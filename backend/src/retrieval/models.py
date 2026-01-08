from pydantic import BaseModel, field_validator, Field, model_validator
from typing import Any, Dict


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
