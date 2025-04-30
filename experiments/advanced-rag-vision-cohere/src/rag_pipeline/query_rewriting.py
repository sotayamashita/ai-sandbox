import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def rewrite_query(original_query: str) -> str:
    prompt = f"""
    Please rewrite the user's question to be more specific and better suited for search engines.
    Original question: {original_query}
    Rewritten question:
    """
    response = gemini_client.models.generate_content(
        model="gemini-1.5-pro", contents=[prompt]
    )
    return response.text.strip()
