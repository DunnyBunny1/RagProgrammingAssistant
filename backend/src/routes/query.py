from fastapi import Depends, HTTPException, APIRouter, Request
from typing import List, Dict, Any
import logging

from sqlmodel import Session, select, text

from backend.src.rag.generator import LlmClient
from backend.src.schemas.requests import QueryRequestBody
from backend.src.schemas.responses import QueryResponse
from backend.src.rag.retriever import SemanticSearchEngine, SemanticSearchResult
from backend.src.schemas.responses import SourceReference
from ragoverflow_shared.models import PostType
from ragoverflow_shared.html_cleaner import clean_and_combine_text

log = logging.getLogger(__name__)

router = APIRouter()

# TODO: Consider refactoring this to "stream" the LLM response back to the frontend
@router.post("/query", response_model=QueryResponse)
async def query(
        request_body: QueryRequestBody,
        request: Request
) -> QueryResponse:
    """
    RAG query: Semantic search + LLM generation.
    TODO: Look into python equivalent of nodejs api-doc library here
    """
    try:
        user_query = request_body.user_query
        duckdb_session: Session = request.app.state.duckdb_session
        semantic_search_engine: SemanticSearchEngine = request.app.state.semantic_search_engine
        llm_client: LlmClient = request.app.state.llm_client

        log.info(f"User query: '{user_query}'")

        # RAG step 1: Information retrieval - launch a semantic search for similar stack overflow posts to the query
        # we want to find the question with the highest semantic similarity
        # use a semantic search against our vectorDB to find the IDs top 3 most similar questions to the incoming query
        search_results: List[SemanticSearchResult] = semantic_search_engine.search_similar_posts(
            query=user_query,
            top_k=3,
        )

        # use our retrieved data to build context for an LLM response
        context: str = build_llm_context(duckdb_session, [search_result.post_id for search_result in search_results])

        # RAG step 2: Synthesis = Produce an LLM response to the query, providing our retrieved data as context
        llm_response = llm_client.generate_answer(
            query=user_query,
            context=context
        )

        # return an HTTP response containing our retrieval-augmented LLM response and the source references
        return QueryResponse(
            llm_response=llm_response,
            sources=format_sources(search_results),
        )



    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


def build_llm_context(
        session: Session,
        post_ids: List[int]
) -> str:
    """
    Fetch posts and build context as just cleaned combined text.
    """
    # TODO: Fix these coments
    # for the matching posts, fetch the post content from DuckDB
    # for each search result, look up the question ID in duckDB to find the top answer (answer with the most net votes)
    # Use the answer text content as context
    # Use the answer's POST ID to derive the source URL
    context_parts = []

    for post_id in post_ids:
        # Fetch the matched post
        post_query = text(f"""
               SELECT post_id, post_type, title, body, tags, net_votes, parent_id
               FROM stackoverflow_posts
               WHERE post_id = {post_id}
           """)

        post = session.exec(post_query).fetchone()

        if not post:
            log.error(f"Unable to find duckDB post with {post_id=} that is present in the vector database")
            raise HTTPException(status_code=500, detail="Internal server error")

        # Convert the post to a dict
        post_dict: Dict[str, Any] = dict(post._mapping)
        # Get cleaned content for this post
        content = clean_and_combine_text(post_dict)

        # If it's a question, try to add the top answer
        if post_dict['post_type'] == PostType.QUESTION.value:
            answer_query = text(f"""
                   SELECT body
                   FROM stackoverflow_posts
                   WHERE post_type = {PostType.ANSWER.value}
                     AND parent_id = {post_id}
                   ORDER BY net_votes DESC
                   LIMIT 1
               """)

            answer = session.exec(answer_query).fetchone()
            if answer:
                answer_content = clean_and_combine_text({'body': answer.body})
                content += f"\n\n{answer_content}"

        # If it's an answer, try to add the parent question
        elif post_dict['post_type'] == PostType.ANSWER.value and post_dict.get('parent_id', ''):
            question_query = text(f"""
                   SELECT title, body
                   FROM stackoverflow_posts
                   WHERE post_id = {post_dict['parent_id']}
                     AND post_type = {PostType.QUESTION.value}
               """)

            question = session.exec(question_query).fetchone()
            if question:
                question_content = clean_and_combine_text({
                    'title': question.title,
                    'body': question.body
                })
                # Store question info for source formatting
                post_dict['question_title'] = question.title
                # Prepend question before answer
                content = f"{question_content}\n\n{content}"

        context_parts.append(content)

    # Just join all the cleaned content together
    context = "\n\n---\n\n".join(context_parts)

    return context


def format_sources(search_results: List[SemanticSearchResult]) -> List[SourceReference]:
    return [
        SourceReference(
            url=f"https://stackoverflow.com/{get_stackoverflow_url_path(post.post_type)}/{post.post_id}",
            cosine_similarity_score=post.similarity_score,
        )
        for post in search_results
    ]


def get_stackoverflow_url_path(post_type: int) -> str:
    return "questions" if post_type == PostType.QUESTION.value else "a"
