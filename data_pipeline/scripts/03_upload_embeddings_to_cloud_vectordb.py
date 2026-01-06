"""
Uploads embeddings from Parquet file to Pinecone vector database.
Reads embeddings generated in step 2, uploads them in batches to Pinecone for semantic search.
Assumes the Pinecone index already exists.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from tqdm import tqdm

from data_pipeline.models import PineconeVector
from data_pipeline.utils.logging_config import setup_logging

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env", verbose=True)

# the index name is the name of the collection where our vectors will live
# each vector will have its primary key ID, its embedding, and optional metadata key-value pairs
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# api key for interacting with the pinecone cloud server ¬
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

UPLOAD_BATCH_SIZE = 100  # the number of vectors to upload together at a time

MAX_HTTP_RETRIES = 3

DATA_DIR_PATH = Path(__file__).parent.parent / "data"
EMBEDDINGS_INPUT_PATH = DATA_DIR_PATH / "embeddings" / "post_embeddings_BAAI-bge-small-en-v1.5.parquet"

log = setup_logging(__file__)


def connect_to_index(pc: Pinecone):
    """
    Connect to an existing Pinecone index. Exit if index does not exist.

    :param pc: Pinecone client instance
    :return: Pinecone Index object
    """
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        log.error(f"Index '{PINECONE_INDEX_NAME}' does not exist")
        log.error(f"Available indexes: {existing_indexes}")
        log.error(f"Please create the index first before running this script")
        sys.exit(1)

    log.info(f"Connecting to existing index '{PINECONE_INDEX_NAME}'...")
    index = pc.Index(PINECONE_INDEX_NAME)

    # log stats on our index - displays info such as vector dimension, total vector count, distance metric, etc
    log.info(f"Current index stats: {index.describe_index_stats()}")

    return index


def upload_batch_with_retry(index, vectors: List[PineconeVector], batch_num: int) -> bool:
    """
    Upload a batch of vectors to Pinecone with exponential backoff retry logic.
    Retries help handle transient network failures or rate limiting.

    :param index: Pinecone index object
    :param vectors: List of Pinecone vectors
    :param batch_num: Batch number for logging purposes
    :return: True if upload succeeded, False if all retries failed
    """

    # convert vectors to Pinecone's expected tuple format: (id, values, metadata_dict)
    vectors: List[Tuple[str, List[float], Dict[str, Any]]] = [vector.to_pinecone_format() for vector in vectors]

    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        try:
            # Attempt to upsert the batch
            # "upsert" = insert new vectors or update existing ones (by ID)
            index.upsert(vectors=vectors)
            return True

        except Exception as e:
            if attempt < MAX_HTTP_RETRIES:
                # Calculate exponential backoff wait time: 2^0=1s, 2^1=2s, 2^2=4s
                wait_time = 2 ** (attempt - 1)
                log.warning(
                    f"Batch {batch_num} upload failed (attempt {attempt}/{MAX_HTTP_RETRIES}): {e}"
                )
                log.warning(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # All retries exhausted
                log.error(f"Batch {batch_num} failed after {MAX_HTTP_RETRIES} attempts: {e}")
                return False

    return False


def upload_all_embeddings(df: pd.DataFrame, index) -> bool:
    """
    Upload all embeddings from DataFrame to Pinecone in batches.

    :param df: DataFrame containing embeddings and metadata
    :param index: Pinecone index object
    :return: True if all batches uploaded successfully, False otherwise
    """
    log.info(f"Starting upload of {len(df):,} vectors (batch size: {UPLOAD_BATCH_SIZE})...")

    total_batches = (len(df) + UPLOAD_BATCH_SIZE - 1) // UPLOAD_BATCH_SIZE
    failed_batches = []

    # Process DataFrame in batches
    for i in tqdm(range(0, len(df), UPLOAD_BATCH_SIZE), desc="Uploading vectors to PineconeDB..."):
        batch_num = i // UPLOAD_BATCH_SIZE + 1
        df_batch = df.iloc[i:i + UPLOAD_BATCH_SIZE]

        # Convert DataFrame rows to Pinecone format
        vectors: List[PineconeVector] = [PineconeVector.from_dataframe_row(row) for _, row in df_batch.iterrows()]

        # Upload with retry logic
        if not upload_batch_with_retry(index, vectors, batch_num):
            failed_batches.append(batch_num)

    # Report results
    if failed_batches:
        log.error(f"Failed to upload {len(failed_batches)} batches: {failed_batches}")
        return False

    log.info(f"Successfully uploaded all {len(df):,} vectors in {total_batches} batches")

    # Wait for index to process updates, then verify final stats
    time.sleep(2)
    final_stats = index.describe_index_stats()
    log.info(f"After upload, index stats: {final_stats}")

    return True


def main() -> int:
    """
    Main pipeline: load embeddings from Parquet, upload to Pinecone.

    :return: 0 on success, 1 on failure
    """
    try:
        # validate required env vars are provided
        if not PINECONE_INDEX_NAME:
            raise ValueError("PINECONE_INDEX_NAME environment variable is not set")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is not set")

        # Load embeddings from Parquet file
        df = pd.read_parquet(EMBEDDINGS_INPUT_PATH)

        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)

        # Connect to existing index (exits if index doesn't exist)
        index = connect_to_index(pc)

        # Upload all embeddings
        if not upload_all_embeddings(df, index):
            log.error("Upload failed - some batches could not be uploaded")
            return 1

        log.info("Upload to Pinecone complete - ready for semantic search")
        return 0

    except Exception as e:
        log.error(f"Error in main: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
