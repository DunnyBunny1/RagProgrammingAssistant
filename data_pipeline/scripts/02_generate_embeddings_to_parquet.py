"""
Generates embeddings for Stack Overflow posts stored in DuckDB.
Reads posts, cleans and truncates text to fit model limits, generates
384-dimensional embeddings, and saves to Parquet format.
"""

import logging
import sys
import warnings
from pathlib import Path
import pandas as pd
from bs4 import MarkupResemblesLocatorWarning
from sentence_transformers import SentenceTransformer
from sqlmodel import create_engine, Session, text
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

from data_pipeline.models import StackOverflowPost, PostType
from data_pipeline.utils.html_cleaner import clean_html

# suppress the beautifulsoup warning about the text content - this warning pops up if the textual content
# we are parsing resembles a filename / pathS since we are only parsing textual content from in-memory data structures,
# the warning can be suppressed
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# BGE model name = "BAAI General Embedding", a pretrained embedding generation model
# "General" = model is designed to work across many domains (not just code, not just medical text, etc.)
# "Embedding" = the model's purpose is to convert text to dense vectors that capture that text's semantic meaning
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# the max number of tokens that input text can contain
EMBEDDING_MODEL_INPUT_MAX_TOKENS = 512

# the number of posts for which to generate embeddings at a time
BATCH_SIZE = 1_000

DATA_DIR_PATH = Path(__file__).parent.parent / "data"
DUCKDB_PATH: Path = DATA_DIR_PATH / "duckdb" / "stackoverflow_analysis.db"
EMBEDDINGS_OUTPUT_PATH = DATA_DIR_PATH / "embeddings" / f"post_embeddings_{EMBEDDING_MODEL_NAME.replace('/', '-')}.parquet"

# configure logging to write logs to an output file and std-out
file_handler = logging.FileHandler(
    filename=Path(__file__).parent.parent / "logs" / f"{Path(__file__).stem}.log",
    mode="w",
)
stream_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(
    handlers=[file_handler, stream_handler],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)

log = logging.getLogger(__name__)


def count_tokens(text: str, tokenizer) -> int:
    """
    Returns the # of tokens when the given text is encoded by the given tokenizer
    """
    return len(tokenizer.encode(text, add_special_tokens=True))


def truncate_text_to_fit_within_token_limit(text: str, tokenizer) -> str:
    """
    Truncates the given text to fit within MAX_TOKENS limit

    :param text: Text to truncate
    :param tokenizer: HuggingFace tokenizer
    """
    # encode the given text to tokens
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # if the text is within the token limit, return the text as-is (no truncation)
    if len(tokens) <= EMBEDDING_MODEL_INPUT_MAX_TOKENS:
        return text

    # if the text has too many tokens, truncate the tokens, and decode tokens back to text
    truncated_tokens = tokens[:EMBEDDING_MODEL_INPUT_MAX_TOKENS]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True)


def load_posts_from_duckdb(session: Session) -> pd.DataFrame:
    """
    Load all posts from DuckDB into a DataFrame.

    :param session: SQLModel database session
    :return: DataFrame with post data
    """
    log.info("Loading posts from DuckDB...")
    # Load all posts
    query = f"""
        SELECT 
            post_id,
            post_type,
            title,
            body,
            net_votes,
            tags
        FROM {StackOverflowPost.__tablename__}
    """

    df = pd.read_sql(query, session.connection())
    log.info(f"Loaded {len(df):,} posts")

    return df


def prepare_post_data_for_embeddings(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """
    Prepare post data for embeddings.
    Computes the post content by combining the title and body, strips HTML from within the content, and truncates the
    content to fit within the embedding model's input token limit

    :param df: DataFrame with raw post data
    :param tokenizer: HuggingFace tokenizer
    :return: DataFrame with prepared_content and token_count columns
    """
    log.info("Preparing posts for embedding...")

    def clean_and_combine_text(row):
        # combine the cleaned post title (if present) with the cleaned post body 
        # for posts that do not have a title, just clean the body 
        title = clean_html(row.get('title', ''))
        body = clean_html(row.get('body', ''))
        return f"{title}\n\n{body}" if title else body

    log.info("Cleaning HTML and combining text on all posts...")
    df['full_cleaned_content'] = df.apply(clean_and_combine_text, axis=1)

    # Count tokens
    log.info("Counting tokens on all posts...")
    df['token_count'] = df['full_cleaned_content'].apply(lambda x: count_tokens(x, tokenizer))

    # Truncate posts
    log.info(f"Truncating all posts to {EMBEDDING_MODEL_INPUT_MAX_TOKENS} tokens...")
    df['prepared_content'] = df['full_cleaned_content'].apply(
        lambda x: truncate_text_to_fit_within_token_limit(x, tokenizer))
    df['was_truncated'] = df['token_count'] > EMBEDDING_MODEL_INPUT_MAX_TOKENS

    # Log truncation stats
    posts_truncated = df['was_truncated'].sum()
    pct_truncated = (posts_truncated / len(df)) * 100
    log.info(f"{posts_truncated:,} posts ({pct_truncated:.1f}%) were truncated")

    log.info(f"Prepared {len(df):,} posts for embedding\n")

    return df


def generate_embeddings(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """
    Generate embeddings for all posts in batches.

    :param df: DataFrame with prepared_content column
    :param model: SentenceTransformer model
    :return: DataFrame with embedding column added
    """
    log.info(f"Generating embeddings (batch size: {BATCH_SIZE})...")

    all_embeddings = []
    texts = df['prepared_content'].tolist()

    # Process in batches. This is because the embedding model's underlying neural network processes the batch as a
    # matrix operation on the GPU/CPU, which is much faster than looping
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
        batch = texts[i:i + BATCH_SIZE]
        # Generate embeddings for entire batch as a "batch operation". This is just a performance optimizaiton on
        # the hardware, and each embedding still remains independent
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings)

    df['embedding'] = all_embeddings

    log.info(f"Generated {len(all_embeddings):,} embeddings")
    log.info(f"Embedding dimension: {all_embeddings[0].shape[0]}")

    return df


def save_to_parquet(df: pd.DataFrame):
    """
    Save embeddings and metadata to Parquet file at EMBEDDINGS_OUTPUT_PATH

    :param df: DataFrame with embeddings
    """
    log.info(f"Saving to {EMBEDDINGS_OUTPUT_PATH}...")

    # Convert embeddings to list format for Parquet
    df['embedding'] = df['embedding'].apply(lambda x: x.tolist())

    # Select columns to save
    columns = [
        'embedding',  # the embedding for the post content
        'post_id',  # id of the post - this will be used as the vector ID (primary key)
        # Post metadata columns - these will be stored in our vectorDB for filtering during the vector search
        'post_type',
        'net_votes',
        'tags'
    ]

    df = df[columns]

    # Save to Parquet
    df.to_parquet(EMBEDDINGS_OUTPUT_PATH, index=False)

    file_size_mb = EMBEDDINGS_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    log.info(f"Saved {len(df):,} embeddings ({file_size_mb:.2f} MB)")


def main() -> int:
    """
    Main pipeline: load posts, generate embeddings, save to Parquet.

    :return: 0 on success, 1 on failure
    """
    try:

        # Check if embeddings already exist
        if EMBEDDINGS_OUTPUT_PATH.exists():
            log.warning(f"Detected that embeddings already exist at {EMBEDDINGS_OUTPUT_PATH}")
            return 0

        # Load tokenizer
        log.info(f"Loading tokenizer: {EMBEDDING_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        log.info("Tokenizer loaded\n")

        # Connect to DuckDB and load posts
        engine = create_engine(f"duckdb:///{DUCKDB_PATH}")
        with Session(engine) as session:
            df = load_posts_from_duckdb(session)

        if len(df) == 0:
            log.error("No posts found in database")
            return 1

        # Prepare posts (clean, combine, count tokens, truncate)
        df = prepare_post_data_for_embeddings(df, tokenizer)

        # Load embedding model
        log.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        log.info("Model loaded\n")

        # Test model
        # test_text = "How do I sort a list in Python?"
        # test_embedding = model.encode(test_text)
        # log.info(f"Model test:")
        # log.info(f"  Input: '{test_text}'")
        # log.info(f"  Output shape: {test_embedding.shape}\n")
        # expected output: 384 x 1

        # Generate embeddings
        embeddings_df = generate_embeddings(df, model)

        # Save to Parquet
        save_to_parquet(embeddings_df)

        return 0

    except Exception as e:
        log.error(f"Error in main: {e}", exc_info=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
