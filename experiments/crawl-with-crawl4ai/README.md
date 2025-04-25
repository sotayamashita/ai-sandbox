# Website Crawler with crawl4ai

A powerful website crawling solution that uses the `crawl4ai` library to efficiently extract and save website content as markdown files.

## Features

- 🕸️ **Breadth-First Search (BFS)** - Systematically crawls websites using BFS strategy for comprehensive coverage
- 🔍 **Depth Control** - Limits crawling depth to 2 levels to prevent excessive data collection
- 🤖 **Robots.txt Compliance** - Automatically respects website crawling rules defined in robots.txt
- ⏱️ **Rate Limiting** - Implements 1-2 second delays between requests to minimize server load
- 🔄 **Automatic Retry** - Features retry mechanism (up to 2 retries) for handling transient errors
- 💾 **Response Caching** - Enables caching for improved performance and reduced server load
- 📊 **Crawl Analytics** - Provides detailed summaries of successful and failed page crawls

## Overview

This experiment demonstrates an efficient web crawling solution that:
1. Crawls the [crawl4ai documentation](https://docs.crawl4ai.com/) or any specified website
2. Extracts valuable content from each page
3. Processes and saves the content as markdown files for further use
4. Reports on crawl statistics and performance

## Setup

### Prerequisites

- Python 3.10 or higher
- `uv` package manager (recommended for dependency management)

### Installation

1. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate.fish
```

2. Install dependencies:
```bash
uv sync
```

## Usage

Run the script with:

```bash
uv run src/main.py --start-url=https://example.com
```

### How It Works

The script will:
- Create a `tmp` directory (if it doesn't exist)
- Crawl the specified website following the configured parameters
- Process and save each crawled page as a markdown file in the `tmp` directory
- Print a comprehensive summary of successful and failed pages upon completion

### Configuration Options

You can customize the crawler by modifying the script parameters:
- Start URL: `--start-url` defines the entry point for crawling
- Depth: Modify crawler initialization to change the maximum crawling depth
- Rate limits: Adjust delay parameters to change request frequency
- Retry policy: Configure retry attempts and backoff strategies

## Use Cases

- Knowledge base creation from documentation sites
- Content aggregation for offline reading
- Data collection for training RAG systems like [`experiments/rag-md-duckdb`](../rag-md-duckdb)
- Website archiving and backup

## References

- [unclecode/crawl4ai](https://github.com/unclecode/crawl4ai) - Official crawl4ai library repository
- [Web Crawling Best Practices](https://developers.google.com/search/docs/crawling-indexing/robots/intro) - Google's guidelines for responsible web crawling
