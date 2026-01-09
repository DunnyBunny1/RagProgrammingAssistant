"""Application configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


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
    # TODO: Consider moving data folder to top level
    duckdb_uri: str = "duckdb:////app/data_pipeline/data/duckdb/stackoverflow_analysis.db"

    # LLM config
    anthropic_api_key: str
    # llm_models = ["claude-3-5-haiku-20241022", "gpt4-o-mini", ]

    # Embedding model
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    default_top_k: int = 5

    class Config:
        env_file = Path(__file__).parent.parent.parent / ".env"
        env_file_encoding = "utf-8"
