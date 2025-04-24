from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import duckdb
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from tqdm.auto import tqdm


@dataclass(frozen=True)
class Config:
    db_file: Path = Path("./data/vectors.db")
    db_table_name: str = "vector_data"
    embedding_model_name: str = "intfloat/multilingual-e5-large-instruct"
    embedding_vector_dim: int = 1024
    input_dir: Path = Path("./input")


CONFIG = Config()
console = Console()


def connect_db() -> duckdb.DuckDBPyConnection:
    CONFIG.db_file.parent.mkdir(parents=True, exist_ok=True)
    db_con = duckdb.connect(str(CONFIG.db_file))

    # Load VSS extension
    try:
        console.print("Loading VSS extension...")
        db_con.sql("INSTALL vss;")
        db_con.sql("LOAD vss;")
        # enable on-disk HNSW persistence (required for file-backed DB)
        db_con.execute("SET hnsw_enable_experimental_persistence=true;")
        console.print("[green]VSS loaded.[/green]")
    except Exception as e:
        console.print(f"[yellow]VSS load failed or already loaded: {e}[/yellow]")
        try:
            db_con.sql("LOAD vss;")
            console.print("[green]VSS loaded (potentially already installed).[/green]")
        except Exception as load_e:
            console.print(
                f"[red]VSS load finally failed: {load_e}. Vector search might be slow or fail.[/red]"
            )

    # Create table if not exists
    db_con.execute(
        f"""
        CREATE OR REPLACE TABLE {CONFIG.db_table_name} (
            id INTEGER,
            text VARCHAR,
            embedding FLOAT[{CONFIG.embedding_vector_dim}]
        );
    """
    )

    # Persist a fast HNSW index (built only once, reused later)
    db_con.execute(
        f"CREATE INDEX IF NOT EXISTS {CONFIG.db_table_name}_hnsw "
        f"ON {CONFIG.db_table_name} USING hnsw(embedding);"
    )

    return db_con


def load_embedding_model() -> HuggingFaceEmbeddings:
    console.print(
        f"\nLoading embedding model: [cyan]{CONFIG.embedding_model_name}[/cyan]..."
    )

    try:
        import torch  # type: ignore

        if torch.backends.mps.is_available():
            device = "mps"
            console.print("MPS device found, using MPS.")
        else:
            device = "cpu"
            console.print("MPS not available, using CPU.")
    except ImportError:
        device = "cpu"
        console.print("PyTorch not found or MPS check failed, using CPU.")

    return HuggingFaceEmbeddings(
        model_name=CONFIG.embedding_model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(model: HuggingFaceEmbeddings) -> List[tuple[int, str, np.ndarray]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    rows: List[tuple[int, str, np.ndarray]] = []
    uid = 0

    # Get a list of markdown files
    md_files = list(CONFIG.input_dir.glob("*.md"))

    # Show progress bar for file processing
    for file in tqdm(md_files, desc="Processing markdown files"):
        loader = UnstructuredMarkdownLoader(str(file))
        md_docs = loader.load()
        chunks = splitter.split_documents(md_docs)
        texts = [chunk.page_content for chunk in chunks]

        # Show progress bar for embedding processing
        for text, vec in tqdm(
            zip(texts, model.embed_documents(texts)),
            desc=f"Embedding chunks from {file.name}",
            total=len(texts),
            leave=False,
        ):
            rows.append((uid, text, np.asarray(vec, dtype=np.float32)))
            uid += 1

    return rows


def main() -> None:
    # Connect to database
    console.print("[cyan]Connecting database…[/cyan]")
    db_con = connect_db()

    # Load embedding model
    console.print("[cyan]Loading embedding model…[/cyan]")
    embedding_model = load_embedding_model()

    try:
        # (Re)bu`ild index if table is empty
        if (
            db_con.execute(f"SELECT COUNT(*) FROM {CONFIG.db_table_name}").fetchone()[0]
            == 0
        ):
            console.print("[cyan]Building vector index…[/cyan]")
            rows = build_index(embedding_model)

            # Show progress bar for database insertion
            console.print("[cyan]Inserting into database…[/cyan]")
            batch_size = 100  # Specify batch size
            for i in tqdm(
                range(0, len(rows), batch_size), desc="Inserting into database"
            ):
                batch = rows[i : i + batch_size]
                db_con.executemany(
                    f"INSERT INTO {CONFIG.db_table_name} (id, text, embedding) VALUES (?, ?, ?)",
                    batch,
                )

            console.print(f"[green]Indexed {len(rows)} chunks.[/green]")

        # Search
        def search(query: str) -> None:
            console.print("[cyan]Searching…[/cyan]")
            query_vector = embedding_model.embed_query(query)
            query_vector_np = np.array(query_vector, dtype=np.float32)
            results = db_con.execute(
                f"""
                SELECT id, text, list_cosine_similarity(embedding, ?) AS sim
                FROM {CONFIG.db_table_name}
                ORDER BY sim DESC
                LIMIT 3;
                """,
                (query_vector_np,),
            ).fetchall()

            if not results:
                console.print("[yellow]No matches found.[/yellow]")
            else:
                for rank, (row_id, text, sim) in enumerate(results, start=1):
                    snippet = text.replace("\n", " ")[:300] + (
                        "…" if len(text) > 300 else ""
                    )
                    console.print(
                        f"[bold]{rank}. (ID {row_id})[/bold] sim={sim:.4f}\n   {snippet}"
                    )

        # REPL mode
        while True:
            try:
                query = input("\nQuery (empty line to quit) > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                break

            search(query)

    finally:
        if db_con:
            db_con.close()
            console.print("[green]DB connection closed.[/green]")


if __name__ == "__main__":
    main()
