import os

import numpy as np


def load_embeddings(path: str) -> np.ndarray:
    if os.path.exists(path):
        return np.load(path)
    raise FileNotFoundError(f"Embedding file {path} not found.")


def save_embeddings(path: str, embeddings: np.ndarray) -> None:
    np.save(path, embeddings)


def find_most_similar(query_vector: np.ndarray, doc_vectors: np.ndarray) -> int:
    similarities = np.dot(query_vector, doc_vectors.T)
    return int(np.argmax(similarities))
