"""Application configuration."""
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Application settings for our backend server. Will automatically fetch from our `.env` file or `os.environ`
    """
    app_name: str = "RagOverflow API"
    app_version: str = "0.1.0"

    # config to connect to our pinecone db vector store - fetch from env vars
    pinecone_api_key: str
    pinecone_index_name: str

    # config to connect to duckDB
    duckdb_uri: str = "duckdb:////app/data_pipeline/data/duckdb/stackoverflow_analysis.db"

    # LLM config
    anthropic_api_key: str
    # TODO: Consider adding /list-models API route to display the list of available models
    llm_models: List[str] = ["claude-haiku-4-5", "claude-sonnet-4-5"]

    # Embedding model
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    default_top_k: int = 5

    class Config:
        env_file = Path(__file__).parent.parent.parent / ".env"
        env_file_encoding = "utf-8"
