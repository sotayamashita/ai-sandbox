from pathlib import Path

import fire
from rag_pipeline import (
    answer_generation,
    embedding_model,
    knowledge_document,
    query_retrieval,
    query_rewriting,
    vector_database,
)

OUTPUT_DIR = Path("data/output")
EMBEDDINGS_FILE = OUTPUT_DIR / "doc_embeddings.npy"


def main(pdf_path: str):
    images = knowledge_document.convert_pdf_to_images(pdf_path, OUTPUT_DIR)
    image_paths = [str(OUTPUT_DIR / f"page_{i + 1}.png") for i in range(len(images))]

    if EMBEDDINGS_FILE.exists():
        doc_embeddings = vector_database.load_embeddings(str(EMBEDDINGS_FILE))
    else:
        doc_embeddings = embedding_model.embed_images(image_paths)
        vector_database.save_embeddings(str(EMBEDDINGS_FILE), doc_embeddings)

    while True:
        try:
            query = input("\nQuery (empty line to quit) > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break

        # Disambiguate Query Agent
        clarified_query = query_rewriting.rewrite_query(query)

        # Embed query
        query_vector = embedding_model.embed_query(clarified_query)
        top_index = vector_database.find_most_similar(query_vector, doc_embeddings)
        top_image_path = image_paths[top_index]

        print("Most relevant image:", top_image_path)
        query_retrieval.show_image(top_image_path)

        print("Generating answer...")
        print(answer_generation.generate_answer(clarified_query, top_image_path))


if __name__ == "__main__":
    fire.Fire(main)
