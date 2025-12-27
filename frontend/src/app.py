import logging
from pprint import pformat
import requests
from requests import Response
import streamlit as st


# Set up basic logging infrastructure
logging.basicConfig(
    format="%(filename)s:%(lineno)s:%(levelname)s -- %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

st.write("Hello from StreamLit ;)")

# Send a request to our backend API container
# TODO: Consider extracting "backend-api:8000" (hostname + port) into constants
response: Response = requests.get("http://backend-api:8000/example")

st.write(f"Response: {pformat(response.json())}")
