# Markdown DuckDB RAG

A simple Retrieval-Augmented Generation (RAG) system that processes markdown documents using DuckDB for vector storage and retrieval.

## Features

- 📚 **Markdown Processing** - Automatically loads and processes Markdown files from an input directory
- 🔍 **Semantic Search** - Performs similarity search using vector embeddings to find relevant content
- 🗄️ **DuckDB Integration** - Uses DuckDB with VSS extension for efficient vector storage and retrieval
- 🤖 **Multilingual Support** - Powered by multilingual embedding models for cross-language search
- 📊 **Progress Visualization** - Shows real-time progress with tqdm during indexing and processing
- ⚡ **Hardware Acceleration** - Utilizes GPU/MPS acceleration when available for faster embeddings
- 🔄 **Interactive REPL** - Simple command-line interface for querying the knowledge base
- 🧩 **Chunk Management** - Intelligently splits documents into overlapping chunks for better context

## Setup

### Prepare markdown

1. Create an input directory:

```bash
mkdir -p input
```

2. Add your markdown files to this directory. You can:

   - Manually copy existing markdown files
   - Generate markdown from websites using tools [`experiments/crawl-with-crawl4ai`](../crawl-with-crawl4ai)
   - Create new markdown documents with your content

3. Ensure your markdown files follow standard markdown syntax for optimal processing.

### Using Local Python Environment

1. Create and activate virtual environment:

```bash
uv venv
source .venv/bin/activate.fish
```

2. Install dependencies:

```bash
uv sync
```

3. Run the application:

```bash
# The first run will build the vector database from markdown files in input/
uv run src/main.py
```

### Using Docker

Build and run the service with Docker Compose:

```bash
docker compose up --build
```

## Tech Stacks

RAG Processing Stages

### 1. Indexing

#### 1.1. Load Documents

- 📚 **LangChain - UnstructuredMarkdownLoader**
  - Function: Loads and parses Markdown files from input directory
  - Features: Supports various Markdown formats while preserving metadata

#### 1.2. Split Documents

- 🧩 **LangChain - RecursiveCharacterTextSplitter**
  - Function: Splits long texts into chunks with overlap
  - Features: Maintains context through adjustable chunk size and overlap settings
  - Alternative: [`spaCy`](https://spacy.io/)

#### 1.3. Embedding

- 🧠 **HuggingFace Embeddings**
  - Model: [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
  - Function: Generates high-quality 1024-dimensional embeddings with multilingual support
  - Benchmark: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- 🚀 **PyTorch**
  - Function: Provides hardware acceleration (GPU/MPS) for embedding model operations
  - Features: Automatically leverages Apple Silicon MPS acceleration when available

#### 1.4. Store Documents

- 📊 **DuckDB with VSS Extension**
  - Function: Lightweight, zero-configuration analytical database for vector storage and search
  - Features:
    - `hnsw` indexing: Fast approximate nearest neighbor search for embeddings
    - `list_cosine_similarity`: Efficient similarity calculations between vectors
  - Alternative: [Chroma](https://github.com/chroma-core/chroma), [Weaviate](https://github.com/weaviate/weaviate)

  | Feature                         | **DuckDB + VSS Extension**                                                  | **Weaviate**                                                              | **Faiss**                                                     |
  | :------------------------------ | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------ | :------------------------------------------------------------ |
  | **Overall**                     | Lightweight all-in-one SQL + vector option for local analytics or batch RAG | Fully-featured vector DB with hybrid search & production features         | Fast, flexible library for custom similarity-search pipelines |
  | **Vector Storage**              | Yes — embeddings stored in `ARRAY` / `LIST` columns inside ordinary tables | Yes — objects and their vectors are stored together                      | Yes (vectors only) — no metadata storage                    |
  | **Document Storage**            | Yes — any SQL table can hold full documents & metadata                     | Yes — object schema includes properties + vector                         | No — keep docs/metadata in an external store                |
  | **Vector indexing**             | HNSW via `CREATE INDEX … USING vss_hnsw`                                   | Built-in Flat, HNSW, dynamic index                                        | Many: HNSW, IVF-PQ, Flat, OPQ, etc.                           |
  | **Similarity search**           | k-NN with cosine, L2, etc. via SQL (`ORDER BY distance`)                    | Vector k-NN and hybrid BM25 + vector                                      | Core API for k-NN with multiple distance metrics              |
  | **Filtering**                   | Standard SQL `WHERE`, joins, window functions                               | Rich boolean & range filters on metadata                                  | Limited — usually filter results externally                  |
  | **Scalability**                 | Scales with host resources; embeddable in any process                       | Cloud-native, horizontal sharding, multi-tenancy                          | Scales as a library you embed; cluster logic is up to you     |
  | **Configuration**               | Zero-config: single shared-library or Python package                        | Runs as a service (Docker/K8s, managed cloud)                             | Linked/installed as a library in application code             |
  | **Persistence**                 | Yes — single DuckDB file or MotherDuck cloud                               | Yes — durable storage back-ends (Badger, RocksDB, etc.)                  | Manual — call `faiss.write_index` / `read_index`            |

### 2. Retrival

#### 2.1. Semantic Search

- 🔍 **Vector Similarity Search with DuckDB**
  - Function: Calculates similarity between query vectors and stored document vectors
  - Features:
    - `list_cosine_similarity`: Efficient similarity calculations between vectors
    - k-nearest neighbor search for retrieving relevant documents
    - Ranking functionality based on similarity scores

#### System Integration & User Interface

- 🔄 **LangChain**
  - Function: Orchestrates the entire RAG pipeline components
- 📈 **tqdm**
  - Function: Shows progress bars for long-running operations like file processing and embedding generation
  - Features: Improves user experience during indexing and database operations
- 🎨 **Rich**
  - Function: Enhances terminal output with colors and formatting
  - Features: Makes logs and search results more readable and user-friendly

## Configuration

The application uses the following default configuration:
<br/>
See [`src/main.py` -> `class Config`](./src/main.py)

## FAQ

- **How is the vector dimension determined?**

  - The vector dimension (1024) is defined by the specific embedding model used. For `intfloat/multilingual-e5-large-instruct`, the dimension is 1024.

- **What happens when I change the embedding model?**

  - When changing models, you'll need to update the `embedding_vector_dim` in the configuration to match the new model's output dimension.
  - The database table will need to be recreated to accommodate the new vector dimensions.

- **How can I optimize performance?**
  - For large document collections, consider increasing batch sizes for database insertion
  - Enable hardware acceleration (GPU/MPS) by ensuring PyTorch is properly installed
  - Adjust chunk size and overlap based on your specific content and query patterns

## References

- [labdmitriy/llm-rag](https://github.com/labdmitriy/llm-rag)
- [DuckDB Documentation](https://duckdb.org/docs/)
- [DuckDB VSS Extension](https://github.com/duckdb/duckdb_vss)
- [HuggingFace E5 Models](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [tqdm Progress Bar](https://github.com/tqdm/tqdm)
