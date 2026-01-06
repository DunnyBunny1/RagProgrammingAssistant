from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()

# TODO: Look into creating config with child class of pydantic `BaseSettings` configured w/ a `.env` file
@app.get("/")
def hello():
    return PlainTextResponse("Hello World from FastAPI")


@app.get("/example")
def example():
    return {"message": "Welcome to an example page!"}


@app.get("/health_check")
def health_check():
    """
    Health check to ensure the server is running successfully. Returns the message "OK"
    with the "200 OK" status code
    """
    return PlainTextResponse("OK", status_code=200)



if __name__ == "__main__":
    import uvicorn
    # run our app, with hot reloading enabled, on port 8000
    # listen on all network interfaces (0.0.0.0) so that the server is accessible from our host machine
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
