import os

from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_answer(question: str, image_path: str) -> str:
    prompt = f"""
    Answer the question based on the following image. Don't use markdown. Please provide enough context for your answer.
    Question: {question}
    """
    image = Image.open(image_path)
    response = gemini_client.models.generate_content(
        model="gemini-1.5-flash", contents=[prompt, image]
    )
    return response.text
