from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter()


@router.get("/health_check")
def health_check():
    """
    Health check to ensure the server is running successfully. Returns the message "OK"
    with the "200 OK" status code
    """
    return PlainTextResponse("OK", status_code=200)
