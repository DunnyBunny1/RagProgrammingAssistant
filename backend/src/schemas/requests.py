from pydantic import BaseModel


class QueryRequestBody(BaseModel):
    """
    Format for the POST request body to the `/query` endpoint
    """
    user_query: str
