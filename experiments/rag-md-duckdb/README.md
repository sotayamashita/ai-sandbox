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

### RAG Processing Stages

#### 1. Markdown Ingestion
- 📚 **LangChain - UnstructuredMarkdownLoader**
  - Function: Loads and parses Markdown files from input directory
  - Features: Supports various Markdown formats while preserving metadata

#### 2. Text Chunking
- 🧩 **LangChain - RecursiveCharacterTextSplitter**
  - Function: Splits long texts into chunks with overlap
  - Features: Maintains context through adjustable chunk size and overlap settings
  - Alternative: [`spaCy`](https://spacy.io/)

#### 3. Embedding Generation
- 🧠 **HuggingFace Embeddings**
  - Model: [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
  - Function: Generates high-quality 1024-dimensional embeddings with multilingual support
  - Benchmark: [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- 🚀 **PyTorch**
  - Function: Provides hardware acceleration (GPU/MPS) for embedding model operations
  - Features: Automatically leverages Apple Silicon MPS acceleration when available

#### 4. Vector Storage & Retrieval
- 📊 **DuckDB with VSS Extension**
  - Function: Lightweight, zero-configuration analytical database for vector storage and search
  - Features:
    - `hnsw` indexing: Fast approximate nearest neighbor search for embeddings
    - `list_cosine_similarity`: Efficient similarity calculations between vectors
  - Alternative: [Chroma](https://github.com/chroma-core/chroma), [Weaviate](https://github.com/weaviate/weaviate)

#### 5. System Integration & User Interface
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
See: `main.py class Config:`

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

- [DuckDB Documentation](https://duckdb.org/docs/)
- [DuckDB VSS Extension](https://github.com/duckdb/duckdb_vss)
- [HuggingFace E5 Models](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [tqdm Progress Bar](https://github.com/tqdm/tqdm)
