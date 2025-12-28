from typing import List, Dict, Any, Set
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from pathlib import Path
from data_pipeline.models import StackOverflowTag
import logging
import sys
import duckdb
import pandas as pd
from duckdb import DuckDBPyConnection

DATA_DIR_PATH = Path(__file__).parent.parent / "data"

# Paths to the XML files containing Stack Overflow data dump Tags & posts
TAGS_XML_PATH: Path = DATA_DIR_PATH / "raw_xml_data" / "tags.xml"
POSTS_XML_PATH: Path = DATA_DIR_PATH / "raw_xml_data" / "posts.xml"

# Path to the parquet data folder, storing parsed XML
PARQUET_DIR_PATH: Path = DATA_DIR_PATH / "parquet_data"

DUCKDB_PATH : Path = DATA_DIR_PATH / "duckdb" / "stackoverflow_analysis.db"

# Create handlers to write output to a log file & std-err
file_handler = logging.FileHandler(
    filename=Path(__file__).parent.parent / "logs" / f"{Path(__file__).stem}.log",
    mode="w",
)
stream_handler = logging.StreamHandler(stream=sys.stdout)

# Setup a logger w/ our handlers
logging.basicConfig(
    handlers=[file_handler, stream_handler],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)

log = logging.getLogger(__name__)


def load_tags_to_duck_db(db_connection: DuckDBPyConnection):
    """
    Parse the tag XML, perform validation, and load to a parquet file.
    Reads this parquet file into DuckDB
    The tags file is approx 5 MB and ~70k rows, so it can be processed in one big batch.
    Raises any exceptions if they occur
    Returns true on success
    """
    log.info("Loading tags to parquet file...")

    # Keep track of the number of valid / invalid XML rows
    success_count: int = 0
    failure_count: int = 0

    # Parse our XML tags data into an element tree
    tree: ET = ET.parse(TAGS_XML_PATH)

    # Extract the root element. Each child element is a `row`, with data stored in key-val pairs
    root: Element = tree.getroot()
    rows: List[Element] = root.findall("row")

    tags: Set[StackOverflowTag] = set()

    for idx, row in enumerate(rows):
        try:
            # Extract the id, name, count from each tag into a dict
            tag_dict: Dict[str, Any] = {
                "id": int(row.get("Id")),
                "tag_name": row.get("TagName"),
                "count": int(row.get("Count")),
            }

            # Validate all tags against our data model. If any tag fails,
            # log an warning and conintnue. If more than 3 tags fail,
            # log an error and abort the program

            tags.add(StackOverflowTag.model_validate(tag_dict))
        except Exception as e:
            log.warning(f"On {idx=}, {tag_dict} failed validation w/ error {e}")
            failure_count += 1

            if failure_count == 3:
                log.error("Max failure count reached, aborting..")
                exit(0)

            continue
        success_count += 1

    # Dump the tags data to a parquet file
    parquet_file_path: Path = PARQUET_DIR_PATH / "stackoverflow_tags.parquet"

    log.info(
        f"After parsing tags {success_count} / {success_count + failure_count} rows processed successfully\n"
        f"Loading tags data to parquet file at {parquet_file_path}..."
    )

    # Convert tags to records
    records: Set[Dict[str, Any]] = [tag.model_dump(mode="python") for tag in tags]

    # Add records to a pandas dataframe and export to a parquet file
    df: pd.DataFrame = pd.DataFrame(records)
    df.to_parquet(path=parquet_file_path, index=False)

    # After creating parquet
    db_connection.execute(f"""
        INSERT INTO stackoverflow_tags 
        SELECT * FROM read_parquet('{parquet_file_path}')
    """)
    return True  # Success indicator


def load_posts_to_duck_db(
        db_connection: DuckDBPyConnection
):
    """
    Parse the posts XML, perform validation, and load to DuckDB
    The posts XML file is 100 GB and 60,000,000 rows, so processes in chunks of 10k lines
    Returns true on success
    """
    # TODO: Impl
    return False


def setup_db_schemas(db_connection: DuckDBPyConnection):
    """
    Create DB tables for tags and posts, respectively,if they
    do not exist
    """
    log.info("Creating database schemas...")

    # Base the schemas off the data models in models.py
    db_connection.execute(
        """
        CREATE TABLE IF NOT EXISTS stackoverflow_tags
        (
            id
            INTEGER
            PRIMARY
            KEY,
            tag_name
            VARCHAR
            UNIQUE
            NOT
            NULL,
            count
            INTEGER
            NOT
            NULL,
        )
        """
    )

    # Add an index on tag_name
    db_connection.execute("CREATE INDEX IF NOT EXISTS idx_tag_name ON stackoverflow_tags (tag_name)")

    # TODO: Add posts create table statement

    log.info("DB schemas intiialized!")


def main() -> int:
    """
    Load tags and posts (from XML data dumps) into duckDB, with
    resuming at any checkpoints where we left off.
    """
    # Create a persistent database DB connection
    with duckdb.connect(DUCKDB_PATH) as db_connection:
        try:
            # Setup the tables schemas for our DB
            setup_db_schemas(db_connection)

            # Load the stack over flow tags into DuckDB, w/ all necessary data validation
            if not load_tags_to_duck_db(db_connection):
                log.error("Detected failure when loading tags, exiting...")
                return 1

            # Load the stack over flow posts into DuckDB, w/ all necessary data validation
            # Resume at any checkpoints where we left off
            if not load_posts_to_duck_db(db_connection):
                log.error(
                    "Detected failure when loading posts, check logs and resume later."
                )
                return 1

        # Catch any unsuccessufl errors and log an error message
        except Exception:
            log.error("Encountered error in main", exc_info=True)
            return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
