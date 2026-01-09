from backend.src.main import app
from fastapi.responses import PlainTextResponse


@app.get("/health_check")
def health_check():
    """
    Health check to ensure the server is running successfully. Returns the message "OK"
    with the "200 OK" status code
    """
    return PlainTextResponse("OK", status_code=200)
