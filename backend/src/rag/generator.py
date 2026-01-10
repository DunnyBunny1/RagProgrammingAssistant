from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
import logging

log = logging.getLogger(__name__)


# TODO: create interface for LLM client to support multiple model vendors (openAI, gemini, QWEN, deepseek, etc.)
class LlmClient:
    """
    Simple LLM client for generating answers using Claude.
    Uses LangChain to handle the API calls.
    """

    def __init__(self, api_key: str, model: str):
        """
        Initialize the LLM client.

        :param api_key: Anthropic API key
        :param model: Claude model to use
        """
        self.llm = ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=0.0,  # TODO: Consider extracting to config
        )

        self.system_prompt = """You are a helpful programming assistant that answers questions using your knowledge and Stack Overflow content.

        Your task:
        1. Answer the user's programming question accurately
        2. Use the provided Stack Overflow context when it's relevant and helpful
        3. If the Stack Overflow context is incomplete, outdated, or not relevant, explain why, then supplement with your own knowledge. 

        Rules:
        - Include code examples when relevant
        - If you use information from the context, it's implicitly cited (user knows the sources)
        - Keep explanations clear and simple
        
        """

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer to the user's query using the provided context.

        :param query: User's programming question
        :param context: Stack Overflow content (cleaned Q&A pairs)
        :return: Generated answer as a string
        """
        # Build the prompt with context + query
        user_message = f"""Context from Stack Overflow:

        {context}
        
        ---
        
        User Question: {query}
        
        Answer:"""

        # Create messages
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_message)
        ]

        # Call the LLM
        log.info(f"Calling LLM with query: '{query}'")
        response = self.llm.invoke(messages)

        # Extract the text content
        answer = response.content

        log.info(f"Generated answer ({len(answer)} chars)")

        return answer
