"""
Pre-processes Stack Overflow's content dump of posts and hashtags (XML format) by parsing, validating, and filtering
each post / hashtag one-by-one, and then loading the dump data to DuckDB, where it can be analyzed using SQL
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Iterator, Tuple
from xml.etree.ElementTree import Element

from sqlmodel import SQLModel, create_engine, Session, text
from tqdm import tqdm

from ragoverflow_shared.models import StackOverflowTag, StackOverflowPost, PostType
from ragoverflow_shared.logging_config import setup_logging

# path to the `data_pipeline/data` directory where we will store raw, semi-processed, and fully processed data
DATA_DIR_PATH = Path(__file__).parent.parent / "data"

# paths to the XML files containing Stack Overflow data dump Tags & posts
TAGS_XML_PATH: Path = DATA_DIR_PATH / "raw_xml_data" / "tags.xml"
POSTS_XML_PATH: Path = DATA_DIR_PATH / "raw_xml_data" / "posts.xml"

# path to our duckDB SQL database containing our SQL files
DUCKDB_PATH: Path = DATA_DIR_PATH / "duckdb" / "stackoverflow_analysis.db"

log = setup_logging(__file__)

# The score threshold for the top 250k posts (since we are only allowed 2 GB (~250-300k vectors)
# of cloud storage in PineconeDB free tier). computed in `data_exploration.ipynb`
POST_NET_VOTES_THRESHOLD: int = 65
POST_BATCH_SIZE = 5_000  # amount of posts to accumulate in memory before writing to database


def load_tags_to_duck_db(session: Session):
    """
    Parse the tag XML, perform validation, and load to DuckDB
    The tags file is approx 5 MB and ~70k rows, so it can be processed in one big batch.
    Raises any exceptions if they occur
    Returns true on success
    """
    log.info("Loading tags to DuckDB...")

    # Check if tags have already been loaded to DuckDB, skipping loading if so
    existing_tag_count: int = session.exec(text(
        f"""
        SELECT COUNT(*) FROM {StackOverflowTag.__tablename__}
        """
    )).scalar()
    if existing_tag_count > 0:
        log.info(f"Detected that tags have already been loaded, {existing_tag_count=}. Skipping...")
        return True

    # Keep track of the number of valid / invalid XML rows according to our tags data model
    success_count: int = 0
    failure_count: int = 0

    # Parse our XML tags data into an element tree
    tree: ET = ET.parse(TAGS_XML_PATH)

    # Extract the root element. Each child element is a `row`, with data stored in key-val pairs
    root: Element = tree.getroot()
    rows: List[Element] = root.findall("row")

    tags: List[StackOverflowTag] = []

    for idx, row in enumerate(rows):
        try:
            # Extract the id, name, count from each tag into a dict
            tag_dict: Dict[str, Any] = {
                "id": int(row.get("Id")),
                "tag_name": row.get("TagName"),
                "count": int(row.get("Count")),
            }

            # Validate all tags against our data model
            tags.append(StackOverflowTag.model_validate(tag_dict))
            success_count += 1
        except Exception as e:
            # If any tag fails, log a warning and conintnue.
            log.warning(f"On {idx=}, {tag_dict} failed validation w/ error {e}", exc_info=True)
            failure_count += 1

            # If more than 3 tags fail, log an error and abort the program
            if failure_count == 3:
                log.error("Max failure count reached, aborting..")
                exit(0)

    log.info(
        f"After parsing tags, {success_count} / {success_count + failure_count} rows processed successfully\n"
    )

    # add all the tags to duckDB
    session.add_all(tags)
    session.commit()
    return True


def load_posts_to_duck_db(
        session: Session
):
    """
    Parses the posts XML, performs validation, and loads to DuckDB
    The `posts.xml` file is 100 GB and 60,000,000 rows, so processes in batches
    Returns true on success, false on failure
    """
    log.info("Loading posts to DuckDB...")

    # Check if posts have already been loaded to DuckDB, skipping loading if so
    # TODO: Consider adding behavior here to resume from a pre-set checkpoint
    existing_post_count: int = session.exec(text(
        f"""
        SELECT COUNT(*) FROM {StackOverflowPost.__tablename__}
        """
    )).scalar()
    if existing_post_count > 0:
        log.info(f"Detected that posts have already been loaded, {existing_post_count=}. Skipping...")
        return True

    # keep track of the amount of valid / invalid posts according to our posts data model
    success_count: int = 0
    failure_count: int = 0

    current_batch: List[StackOverflowPost] = []  # the posts in the current batch

    # get an iterator that reads XML elements one-by-one, instead of loading all 60 million XML rows file all at once
    # `events=('end',)` means that the iterator will be "notified" each time an XML element is fully parsed
    posts_iterator: Iterator[Tuple[str, Element]] = ET.iterparse(POSTS_XML_PATH, events=('end',))

    # scroll through each XML element as it is parsed and write to duckDB
    # use the TQDM library to display a progress bar as we process posts
    event: str
    element: Element
    for event, element in tqdm(
            posts_iterator,
            desc="Writing posts to DuckDB",  # label shown on the progress bar
            unit=" posts",
            total=60_000_000,  # estimated amount of posts in our raw XML dump
            miniters=10_000,  # update display every 10k posts
    ):
        # skip any elements that aren't` <row>` tags (ex. the root element)
        if element.tag != 'row':
            continue
        try:
            # extract the post info into a dict
            post_dict: Dict[str, Any] = {
                "post_id": int(element.get("Id")),
                "post_type": int(element.get("PostTypeId")),
                "title": element.get("Title", ""),
                "body": element.get("Body", ""),
                "tags": element.get("Tags", ""),
                "net_votes": int(element.get("Score", 0)),
                "parent_id": int(element.get("ParentId")) if element.get("ParentId") else None,
            }

            # validate the post against our data model
            post = StackOverflowPost.model_validate(post_dict)
            success_count += 1

            # skip any posts that have not reached our score threshold
            if post.net_votes < POST_NET_VOTES_THRESHOLD:
                continue

            # skip any posts that are not questions or answers
            if post.post_type not in [PostType.QUESTION.value, PostType.ANSWER.value]:
                continue

            current_batch.append(post)  # add the post to the current batch

            # when the batch is full, write the batches to duckDB and reset the current batch
            if len(current_batch) >= POST_BATCH_SIZE:
                session.add_all(current_batch)
                session.commit()
                current_batch.clear()

        except Exception as e:
            log.warning(
                f"Post {success_count + failure_count} with id: {element.get('id')} failed validation: {e}",
                exc_info=True
            )
            failure_count += 1
            if failure_count >= 3:
                log.error("Max failure count reached, aborting...")
                return False

    # write any remaining posts in the final batch
    if current_batch:
        session.add_all(current_batch)
        session.commit()

    return True


def main() -> int:
    """
    Load tags and posts (from XML data dumps) into duckDB, with
    resuming at any checkpoints where we left off.
    Returns 0 on success, 1 on failure
    """
    try:
        # create a persistent connection to our duckDB database
        engine = create_engine(f"duckdb:///{DUCKDB_PATH}")

        # create the tables and their schemas for our SQLModel classes
        # this auto-generates SQL DDL from the data class specs, then runs the DDL statements
        SQLModel.metadata.create_all(engine)

        # create a session, so we can group multiple SQL write operations into a single transaction
        with Session(engine) as session:
            # load the StackOverflow tags into DuckDB, w/ all necessary data validation
            if not load_tags_to_duck_db(session):
                log.error("Detected failure when loading tags, exiting...")
                return 1

            # Load the StackOverflow flow posts into DuckDB, w/ all necessary data validation
            # TODO: Add functionality to resume at checkpoints where we may leave off
            if not load_posts_to_duck_db(session):
                log.error(
                    "Detected failure when loading posts, check logs and resume later."
                )
                return 1

    # catch any errors and log an error message
    except Exception:
        log.error("Encountered error in main", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
