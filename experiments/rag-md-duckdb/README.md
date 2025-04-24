# rag-markdown-duckdb

## Setup

## Usage

## Teck Stacks

- `torch`
  - Purpose: TODO, write purpose in english around 50 words
  - Verify : `uv run python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"`
- `sentence-transformers`
  - Purpose: TODO, write purpose in english around 50 words
  - Verify : `uv run python -c "import sentence_transformers; print('OK')"`
- `duckdb`
  - Purpose: TODO, write purpose in english around 50 words
    - what is `vss` plugin?
      - TBD, write the answer and usage like when should i enable the plugin and what is the good for?
  - Verify : 
    - `uv run python -c "import duckdb; print(duckdb.__version__)"`
    - `duckdb data/vectors.db "SELECT id, text FROM vector_data;"` required dubck db cli 

## Question

- How to know the `VECTOR_DIM` value?

## References
