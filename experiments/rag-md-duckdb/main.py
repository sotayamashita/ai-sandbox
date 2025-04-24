import duckdb
import argparse
from pathlib import Path
from rich.console import Console
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np # ステップ 24 で必要
# import json # メタデータを使うようになったら必要

# --- 定数 ---
DB_FILE = Path("./data/vectors.db") # DBファイル名を確認
TABLE_NAME = "vector_data"         # テーブル名を確認
VECTOR_DIM = 384                   # 使用するモデルのベクトル次元数
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INPUT_DIR = Path("./input") # Markdown ファイルがあるディレクトリ名を確認

# --- 関数 ---

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query string.")
    return parser.parse_args()

def initialize_database(console):
    """DBファイルに接続し、VSSをロードし、テーブルを作成する。接続オブジェクトを返す"""
    con = None
    try:
        console.print(f"\nConnecting to DB: [cyan]{DB_FILE}[/cyan]")
        # DBファイルが存在するディレクトリを作成（なければ）
        DB_FILE.parent.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(database=str(DB_FILE), read_only=False)
        console.print("[green]DB Connected.[/green]")

        # VSSロード試行
        try:
            console.print("Loading VSS extension...")
            con.sql("INSTALL vss;")
            con.sql("LOAD vss;")
            console.print("[green]VSS loaded.[/green]")
        except Exception as e:
            # すでにインストール済みなどの場合もあるので警告に留める
            console.print(f"[yellow]VSS load failed or already loaded: {e}[/yellow]")
            # ロードだけ再試行
            try:
                con.sql("LOAD vss;")
                console.print("[green]VSS loaded (potentially already installed).[/green]")
            except Exception as load_e:
                # VSSが必須でなければ警告で続行、必須ならエラーにする
                console.print(f"[red]VSS load finally failed: {load_e}. Vector search might be slow or fail.[/red]")

        # テーブル作成
        console.print(f"Creating table '{TABLE_NAME}' if not exists...")
        try:
            # PRIMARY KEY は ID が一意になる場合に設定推奨
            con.execute(f"""
            CREATE OR REPLACE TABLE {TABLE_NAME} (
                id INTEGER,
                text VARCHAR,
                embedding FLOAT[{VECTOR_DIM}]
            );
            """)
            # テーブルスキーマ確認 (任意)
            # schema = con.execute(f".schema {TABLE_NAME}").fetchall()
            # console.print(f"Table Schema for {TABLE_NAME}:\n{schema}")
            console.print(f"Table '{TABLE_NAME}' created/replaced successfully.")
            return con # 成功したら接続オブジェクトを返す
        except Exception as e:
            console.print(f"[bold red]Table creation failed: {e}[/bold red]")
            if con: con.close() # テーブル作成失敗時は接続を閉じる
            return None

    except Exception as e:
        console.print(f"[bold red]DB connection or initialization error: {e}[/bold red]")
        if con: con.close()
        return None

def load_embedding_model(console):
    """Embeddingモデルをロードして返す"""
    console.print(f"\nLoading embedding model: [cyan]{EMBEDDING_MODEL_NAME}[/cyan]...")
    try:
        # MPSが利用可能かチェック (任意だが親切)
        try:
            import torch
            if torch.backends.mps.is_available():
                device = 'mps'
                console.print("MPS device found, using MPS.")
            else:
                device = 'cpu'
                console.print("MPS not available, using CPU.")
        except ImportError:
            device = 'cpu'
            console.print("PyTorch not found or MPS check failed, using CPU.")

        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True} # コサイン類似度を使う場合はTrue推奨
        model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        console.print("[green]Embedding model loaded.[/green]")
        # モデルの次元数をここで確認しても良い
        # test_vec = model.embed_query("test")
        # console.print(f"Model vector dimension check: {len(test_vec)}")
        return model
    except Exception as e:
        console.print(f"[bold red]Failed to load embedding model: {e}[/bold red]")
        return None

# --- ステップ24で追加/修正する関数 ---
def insert_chunk_data(con, console, chunk_id, chunk_text, chunk_vector):
    """単一チャンクのデータをDBに挿入する (ステップ24テスト用)"""
    if not con:
        console.print("[red]DB connection is not valid for insertion.[/red]")
        return False
    if chunk_vector is None:
        console.print("[red]Vector data is missing, cannot insert.[/red]")
        return False

    console.print(f"\n--- Step 24: Inserting single chunk data ---")
    console.print(f"Inserting chunk ID {chunk_id} into DB table '{TABLE_NAME}'...")
    try:
        # --- テストのため、挿入前にテーブルをクリア ---
        console.print(f"Clearing table '{TABLE_NAME}' before insertion (for testing step 24)...")
        con.execute(f"DELETE FROM {TABLE_NAME};")
        console.print(f"Table '{TABLE_NAME}' cleared.")
        # -----------------------------------------

        # データをNumPy配列に変換
        vec_np = np.array(chunk_vector, dtype=np.float32)
        console.print(f"Vector data prepared (shape: {vec_np.shape}, dtype: {vec_np.dtype}).")

        # SQL INSERT文を実行
        con.execute(f"INSERT INTO {TABLE_NAME} (id, text, embedding) VALUES (?, ?, ?)",
                    (chunk_id, chunk_text, vec_np))
        console.print(f"[green]Chunk ID {chunk_id} inserted successfully.[/green]")

        # 挿入後の行数を確認
        res = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()
        if res:
            console.print(f"Current row count in table '{TABLE_NAME}': {res[0]}")
        else:
            console.print("[yellow]Could not verify row count.[/yellow]")
        return True
    except Exception as e:
        console.print(f"[bold red]Failed to insert data for chunk ID {chunk_id}: {e}[/bold red]")
        # エラーの詳細を表示（デバッグ用）
        # import traceback
        # console.print(traceback.format_exc())
        return False
# --- ここまでがステップ24で追加/修正する関数 ---

def main(query):
    console = Console()

    # 1. DB 初期化 (接続、VSSロード、テーブル作成)
    db_con = initialize_database(console)
    if not db_con:
        console.print("[bold red]Database initialization failed. Exiting.[/bold red]")
        return # DB準備ができない場合は終了

    # 2. Embedding モデルロード
    embeddings_model = load_embedding_model(console)
    if not embeddings_model:
        console.print("[bold red]Embedding model loading failed. Exiting.[/bold red]")
        if db_con: db_con.close() # DB接続が開いていれば閉じる
        return # モデル準備ができない場合は終了

    # 3. 最初のMarkdownファイルから最初のチャンクとベクトルを取得
    console.print(f"\nProcessing Markdown files from: [cyan]{INPUT_DIR}[/cyan]")
    md_files = list(INPUT_DIR.glob("*.md"))
    console.print("Found markdown files:", md_files)
    first_chunk = None
    first_chunk_vector = None

    if md_files:
        file_path = md_files[0]
        console.print(f"\nProcessing first file found: [cyan]{file_path}[/cyan]")
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            docs = loader.load()
            if docs:
                console.print("Splitting document into chunks...")
                # チャンクサイズは目的に合わせて調整
                splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
                chunks = splitter.split_documents(docs)
                console.print(f"Number of chunks created: {len(chunks)}")
                if chunks:
                    first_chunk = chunks[0] # 最初のチャンクを取得
                    console.print("\nEmbedding the first chunk...")
                    try:
                        # embed_documents はリストを受け取る
                        chunk_vector_list = embeddings_model.embed_documents([first_chunk.page_content])
                        if chunk_vector_list:
                            first_chunk_vector = chunk_vector_list[0] # 最初のベクトルを取得
                            console.print(f"First chunk embedded. Vector dimension: {len(first_chunk_vector)}")
                        else:
                            console.print("[red]Embedding returned empty list.[/red]")
                    except Exception as e:
                        console.print(f"[bold red]Failed to embed first chunk: {e}[/bold red]")
                else:
                    console.print("[yellow]Document splitting resulted in no chunks.[/yellow]")
            else:
                console.print("[yellow]Loader returned no documents.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error processing file {file_path}: {e}[/bold red]")
    else:
        console.print("[yellow]No markdown files found in input directory.[/yellow]")

    # 4. ステップ 24: 取得した最初のチャンクデータをDBに挿入
    if first_chunk and first_chunk_vector:
        # 挿入処理を呼び出す
        insert_chunk_data(
            con=db_con,
            console=console,
            chunk_id=0, # 最初のチャンクなのでID=0とする（仮）
            chunk_text=first_chunk.page_content,
            chunk_vector=first_chunk_vector
        )
    else:
        # 挿入に必要なデータがない場合
        console.print("\n[yellow]Skipping DB insertion because first chunk or its vector is missing.[/yellow]")

    if embeddings_model:
        console.print(f"\nVectorizing query: '{query}'")
        try:
            query_vector = embeddings_model.embed_query(query)
            query_vector_np = np.array(query_vector, dtype=np.float32)
            console.print(f"Query vector dimension: {len(query_vector)}")

            if query_vector_np is not None: # クエリベクトルがある場合
                try:
                    console.print("\nSearching similar vectors...")
                    results = db_con.execute(
                        f"""
                        SELECT id, text, list_cosine_similarity(embedding, ?) AS similarity
                        FROM {TABLE_NAME} ORDER BY similarity DESC LIMIT 3;
                        """, (query_vector_np,)
                    ).fetchall()
                    console.print("Search results:")
                    if results:
                        context_texts = [row[1] for row in results]
                        context = "\n\n---\n\n".join(context_texts)
                        console.print("\n--- Generated Context (Preview) ---")
                        console.print(context[:300] + "...")
                        console.print("--- End Context Preview ---")
                    else: console.print("  No results.")
                except Exception as e:
                    console.print(f"[bold red]Search failed: {e}[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Query vectorization failed: {e}[/bold red]")
            query_vector_np = None

    # 5. DB 接続クローズ
    if db_con:
        db_con.close()
        console.print("\nDB Connection closed.")
    console.print("\nMain process finished.")

if __name__ == "__main__":
    args = parse_arguments()
    console = Console()
    console.print(f"Received query: [yellow]{args.query}[/yellow]")
    main(args.query)
