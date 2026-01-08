"""
Test semantic search against Pinecone vector database.
Takes a natural language query, embeds it, searches Pinecone for similar posts,
and fetches full content from DuckDB.
"""
from typing import List, Dict, Any
import logging
from pinecone import Pinecone

from ..config import get_config, Config
from sentence_transformers import SentenceTransformer
from .models import SemanticSearchResult

log = logging.getLogger(__name__)


class SemanticSearchEngine:
    """
    Semantic search engine for Stack Overflow posts.
    """

    def __init__(self):
        """TODO: Add docstring"""
        log.info("Initializing semantic search engine...")

        # load our config
        config: Config = get_config()

        # Load embedding model - this will be used to embed text for similarity serach
        log.info(f"Loading embedding model: {config.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(config.embedding_model_name)

        # Connect to Pinecone, allowing us to interact with our pinecone index
        log.info("Connecting to cloud pineconeDB index...")
        pc = Pinecone(api_key=config.pinecone_api_key)
        self.pinecone_index = pc.Index(config.pinecone_index_name)

        log.info("Search engine initialized!")

    def _embed_query(self, query: str) -> List[float]:
        """
        Convert a text query into a 384-dimensional embedding.

        :param query: Natural language search query
        :return: Embedding vector
        """
        embedding = self.embedding_model.encode(query)
        return embedding.tolist()

    def semantic_search_similar_posts(
            self,
            query: str,
            top_k: int,
            filter_dict: Dict[str, Any]
    ) -> List[SemanticSearchResult]:
        """
        Search for posts semantically similar to the query.

        :param query: Natural language search query
        :param top_k: Number of results to return
        :param filter_dict: Dictionary of arguments to filter on vector metadata
        :return: List of matching documents, as a List[ScoredDocument]
        """
        log.info(f"Performing semantic search for top {top_k} similar posts...")

        # embed the query, converting it from natural language (text) to an embedding
        query_embedding = self._embed_query(query)

        # perform a scored vector similarity search of the embedding query against our vector DB
        search_results: List[Dict[str, Any]] = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        ).get("matches", [])

        log.info(f"Found {len(search_results)} similar posts\n")

        # map the raw search results into our data model
        return [
            SemanticSearchResult.from_search_result(search_result)
            for search_result in search_results
        ]
