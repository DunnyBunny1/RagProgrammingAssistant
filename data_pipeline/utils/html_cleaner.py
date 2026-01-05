from bs4 import BeautifulSoup
import re


def clean_html(html_text: str) -> str:
    """
    Remove HTML tags from a string while preserving text content

    :param html_text: Raw HTML string from Stack Overflow post

    :return: Cleaned HTML, as a string

    Ex:
    >>> clean_html("<p>To sort a python list <strong>in-place</strong>, use the <code>sorted()</code> function</p>")
    Output: 'To sort a python list in-place, use the sorted() function
    """

    # Parse HTML
    soup = BeautifulSoup(html_text, 'html.parser')

    # Extract all text (BeautifulSoup handles tag removal)
    text = soup.get_text(separator=' ')

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces → single space
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Replace mulitplme newlines with a double newline

    return text.strip()  # remove any leading / trailing whitespace
