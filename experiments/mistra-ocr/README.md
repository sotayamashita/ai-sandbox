# Mistral OCR Utility

A simple utility for performing OCR on PDF documents using Mistral's API.

## Setup

1. Create and activate virtual environment:
```bash
source .venv/bin/activate.fish
```

2. Install dependencies:
```bash
uv sync
```

3. Create a `.env` file with your Mistral API key:
```bash
cp .env.example .env
```

```bash
MISTRAL_API_KEY=your_api_key_here
```

You can find or create your API key at: https://console.mistral.ai/api-keys

## Usage

Process a PDF file:
```bash
uv main.py path/to/your/document.pdf
```

The tool will:
- Extract text from the PDF using Mistral's OCR service
- Save the content as Markdown in `data/YYYY-MM-DD/document.md`
- Extract and save all images to `data/YYYY-MM-DD/images/` 

## References

- [Mistral OCR](https://mistral.ai/news/mistral-ocr)
- [Mistral OCR and Document Understanding](https://docs.mistral.ai/capabilities/document/)
