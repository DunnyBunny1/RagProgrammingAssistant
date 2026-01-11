from ragoverflow_shared.logging_config import setup_logging
import streamlit as st
import requests
import json

BACKEND_URL = "http://backend-api:8000"

st.set_page_config(
    page_title="RagOverflow",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("Rag Programming Assistant")
st.markdown(
    "Answers technical programming questions using retrieval augmented generation (RAG) over posts on StackOverFlow")

# Query input
query = st.text_input(
    "Ask a programming question:",
    placeholder="Enter your question here, ex. 'How to reverse a string in Python?'",
    key="query_input"
)

# Search button
if st.button("Search", type="primary"):
    if not query:
        st.warning("Please enter a question")
    else:
        with st.spinner("Searching Stack Overflow and generating answer..."):
            try:
                # Call backend API
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"user_query": query},
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    data = response.json()

                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(data["llm_response"])

                    # Display sources
                    st.markdown("### Sources")
                    for i, source in enumerate(data["sources"], start=1):
                        url = source["url"]
                        score = source["cosine_similarity_score"]

                        # Format similarity as percentage
                        similarity_pct = f"{score * 100:.1f}%"

                        st.markdown(f"{i}. [{url}]({url}) - {similarity_pct} match")

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure FastAPI is running on http://localhost:8000")
            except Exception as e:
                st.error(f"Error: {str(e)}")
