import logging
from typing import List, Dict, Any

from pinecone import Pinecone
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ..config import Config

log = logging.getLogger(__name__)


class SemanticSearchResult(BaseModel):
    """
    Represents search results for a scored semantic search performed against our cloud pineconeDB index
    """
    post_id: int = Field(alias="id")
    similarity_score: float = Field(alias="score")
    net_votes: int
    tags: str
    post_type: int

    @classmethod
    def from_search_result(cls, search_result_match: Dict[str, Any]) -> 'SemanticSearchResult':
        """
        Factory method to construct from Pinecone match format.
        Flattens the nested metadata structure.
        """
        return cls(
            id=search_result_match['id'],
            score=search_result_match['score'],
            net_votes=search_result_match['metadata']['net_votes'],
            tags=search_result_match['metadata']['tags'],
            post_type=search_result_match['metadata']['post_type']
        )


class SemanticSearchEngine:
    """
    Semantic search engine for Stack Overflow posts. Serves as a retriever that extracts relevant information from
    the Pinecone cloud pineconeDB index.
    """

    def __init__(self, config: Config):
        """TODO: Add docstring"""
        log.info("Initializing semantic search engine...")

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

    def search_similar_posts(
            self,
            query: str,
            top_k: int = 3,
            metadata_filter_dict: Dict[str, Any] = None
    ) -> List[SemanticSearchResult]:
        """
        Search for posts semantically similar to the query.

        :param query: Natural language search query
        :param top_k: Number of results to return
        :param metadata_filter_dict: Dictionary of arguments to filter on vector metadata
        :return: List of matching documents, as a List[ScoredDocument]
        """
        if not metadata_filter_dict:  # if no filter dict is provided, use an empty filter
            metadata_filter_dict = {}

        log.info(f"Performing semantic search for top {top_k} similar posts...")

        # embed the query, converting it from natural language (text) to an embedding
        query_embedding = self._embed_query(query)

        # perform a scored vector similarity search of the embedding query against our vector DB
        search_results: List[Dict[str, Any]] = self.pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter_dict
        ).get("matches", [])

        log.info(f"Found {len(search_results)} similar posts\n")

        # map the raw search results into our data model
        return [
            SemanticSearchResult.from_search_result(search_result)
            for search_result in search_results
        ]
