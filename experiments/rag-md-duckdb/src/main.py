from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import duckdb
import numpy as np
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console
from tqdm.auto import tqdm

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")


@dataclass(frozen=True)
class Config:
    db_file: Path = Path("./data/vectors.db")
    db_table_name: str = "vector_data"
    embedding_model_name: str = "intfloat/multilingual-e5-large-instruct"
    embedding_vector_dim: int = 1024
    input_dir: Path = Path("./input")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 100
    top_k: int = 3
    llm_model_name: str = "gemini-1.5-flash"
    llm_output_language: str = "Japanese"
    prompt_template_file: Path = Path("./src/prompt_template.txt")


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


def load_llm_model() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CONFIG.llm_model_name, google_api_key=GOOGLE_API_KEY
    )


def load_prompt_template_string() -> str:
    if not CONFIG.prompt_template_file.exists():
        raise FileNotFoundError(
            f"Prompt template file not found: {CONFIG.prompt_template_file}"
        )
    return CONFIG.prompt_template_file.read_text(encoding="utf-8")


def load_embedding_model() -> HuggingFaceEmbeddings:
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.chunk_size, chunk_overlap=CONFIG.chunk_overlap
    )
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
    console.print("[cyan]Connecting database…[/cyan]")
    db_con = connect_db()

    console.print(
        f"Loading embedding model: [cyan]{CONFIG.embedding_model_name}[/cyan]..."
    )
    embedding_model = load_embedding_model()

    console.print(f"Loading LLM: [cyan]{CONFIG.llm_model_name}[/cyan]...")
    llm = load_llm_model()

    console.print(
        f"Loading prompt template from: [cyan]{CONFIG.prompt_template_file}[/cyan]..."
    )
    prompt_template_string = load_prompt_template_string()

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
            batch_size = CONFIG.batch_size  # Specify batch size
            for i in tqdm(
                range(0, len(rows), batch_size), desc="Inserting into database"
            ):
                batch = rows[i : i + batch_size]
                db_con.executemany(
                    f"INSERT INTO {CONFIG.db_table_name} (id, text, embedding) VALUES (?, ?, ?)",
                    batch,
                )

            console.print(f"[green]Indexed {len(rows)} chunks.[/green]")

        # Semantic Search
        def semantic_search(query: str) -> None:
            console.print("[cyan]Searching…[/cyan]")
            query_vector = embedding_model.embed_query(query)
            query_vector_np = np.array(query_vector, dtype=np.float32)
            results = db_con.execute(
                f"""
                SELECT id, text, list_cosine_similarity(embedding, ?) AS sim
                FROM {CONFIG.db_table_name}
                ORDER BY sim DESC
                LIMIT {CONFIG.top_k};
                """,
                (query_vector_np,),
            ).fetchall()

            if not results:
                console.print("[yellow]No matches found.[/yellow]")
            else:
                for rank, (row_id, text, sim) in enumerate(results, start=1):
                    snippet = text.replace("\n", " ")[:140] + (
                        "…" if len(text) > 140 else ""
                    )
                    console.print(
                        f"[bold]{rank}. (ID {row_id})[/bold] sim={sim:.4f}\n   {snippet}"
                    )

        # Retrive context
        def retrieve_context(query: str) -> List[str]:
            console.print("[cyan]Searching…[/cyan]")
            query_vector = embedding_model.embed_query(query)
            query_vector_np = np.array(query_vector, dtype=np.float32)

            results = db_con.execute(
                f"""
                SELECT text, list_cosine_similarity(embedding, ?) AS sim
                FROM {CONFIG.db_table_name}
                ORDER BY sim DESC
                LIMIT {CONFIG.top_k};
                """,
                (query_vector_np,),
            ).fetchall()

            return [row[0] for row in results]

        def generate_answer_with_rag(query: str) -> str:
            context_list = retrieve_context(query)

            if not context_list:
                return "[yellow]No relevant information found.[/yellow]"

            context_str_val = "\n\n".join(context_list)
            query_val = query
            output_language_val = CONFIG.llm_output_language

            try:
                prompt_text = prompt_template_string.format(
                    context_str=context_str_val,
                    query=query_val,
                    output_language=output_language_val,
                )
            except KeyError as e:
                console.print(
                    f"[bold red]KeyError during f-string formatting: {e}. Check placeholder names![/bold red]"
                )
                answer = "[red]Error during prompt formatting (KeyError).[/red]"
                return answer
            except Exception as e:
                console.print(
                    f"[bold red]Error during prompt formatting: {e}[/bold red]"
                )
                answer = "[red]Error during prompt formatting.[/red]"
                return answer

            try:
                response = llm.invoke(prompt_text)
                if hasattr(response, "content") and isinstance(response.content, str):
                    return response.content
                else:
                    console.print(
                        f"[red]Unexpected LLM response type: {type(response)}[/red]"
                    )
                    return "[red]Unexpected LLM response type.[/red]"
            except Exception as e:
                console.print(f"[red]Error during LLM call: {e}[/red]")
                console.print_exception()
                return "[red]Error during LLM call.[/red]"

        # REPL mode
        while True:
            try:
                query = input("\nQuery (empty line to quit) > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                break

            console.print("[cyan]Semantic search...[/cyan]")
            semantic_search(query)

            console.print(
                f"[cyan]Generating answer in {CONFIG.llm_output_language}...[/cyan]"
            )
            answer = generate_answer_with_rag(query)
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(answer)

    finally:
        if "db_con" in locals() and db_con:
            db_con.close()
            console.print("[green]DB connection closed.[/green]")


if __name__ == "__main__":
    main()
