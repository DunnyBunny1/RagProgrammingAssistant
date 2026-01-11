from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from sqlmodel import create_engine, Session

from backend.src.config import Config
from backend.src.rag.generator import LlmClient
from backend.src.rag.retriever import SemanticSearchEngine
from backend.src.routes.health_check import router as health_check_router
from backend.src.routes.query import router as query_router
from ragoverflow_shared.logging_config import setup_logging

# Set up logging for this application
log = setup_logging(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the fastAPI server lifespan.
    Initializes app resources on startup, yields, and then releases resources on shutdown.
    """
    log.info("starting up fastAPI app...")

    # load our app config
    config = Config()

    # initialize our app's semantic search engine
    app.state.semantic_search_engine = SemanticSearchEngine(config=config)

    # initialize our connection to our LLM - by default use the first model name in the config list of LLMs
    app.state.llm_client = LlmClient(api_key=config.anthropic_api_key, model=config.llm_models[0])

    # create a duckDB session and save it to app state
    engine = create_engine(config.duckdb_uri)
    with Session(engine) as session:
        app.state.duckdb_session = session
        # yield to indicate we are done with startup, and the app is now running in a steady state
        yield

    # on app shutdown, log a shutdown message. the context manager will automatically release the duckDB session when
    # we reach here
    log.info("shutting down fastAPI app...")


# create a fastAPI app server and attach the lifespan configuration to it
app = FastAPI(
    title="RagOverflow API",
    description="Semantic search API for Stack Overflow posts",
    version="0.1.0",
    lifespan=lifespan
)

# Add our routers to our app
for router in [query_router, health_check_router]:
    app.include_router(router)


@app.get("/")
def hello():
    return PlainTextResponse("Hello World from FastAPI")


@app.get("/example")
def example():
    return {"message": "Welcome to an example page!"}


if __name__ == "__main__":
    import uvicorn

    # run our app, with hot reloading enabled, on port 8000
    # listen on all network interfaces (0.0.0.0) so that the server is accessible from our host machine
    # TODO: Consider extracting host and port as config settings
    uvicorn.run("backend.src.main:app", host="0.0.0.0", port=8000, reload=True)
