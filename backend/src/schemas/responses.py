from typing import List

from pydantic import BaseModel


class SourceReference(BaseModel):
    url: str
    cosine_similarity_score: float


class QueryResponse(BaseModel):
    """
    Api Response to the `/query` endpoint
    """
    llm_response: str
    sources: List[SourceReference]
