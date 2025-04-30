import os

import cohere
import numpy as np
from dotenv import load_dotenv
from rag_pipeline.knowledge_document import convert_image_to_base64

load_dotenv()
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))


def embed_images(image_paths: list[str]) -> np.ndarray:
    vectors = []
    for path in image_paths:
        response = cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            images=[convert_image_to_base64(path)],
        )
        vectors.append(np.asarray(response.embeddings.float[0]))
    return np.vstack(vectors)


def embed_query(query: str) -> np.ndarray:
    response = cohere_client.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[query],
    )
    return np.asarray(response.embeddings.float[0])
