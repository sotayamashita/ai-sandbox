# Website Crawler with crawl4ai

This experiment demonstrates how to use the `crawl4ai` library to crawl a website and save its content as markdown files.

## Overview

The script in this experiment:
1. Crawls the [crawl4ai documentation](https://docs.crawl4ai.com/)
2. Extracts content from each page
3. Saves the content as markdown files

## Features

- Uses Breadth-First Search (BFS) strategy for crawling
- Limits crawling depth to 2 levels
- Respects robots.txt rules
- Implements rate limiting (with 1-2 second delay between requests)
- Automatic retry mechanism (up to 2 retries)
- Enables caching for better performance
- Provides a summary of successful and failed pages

## Setup

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
uv run src/main.py
```

The script will:
- Create a `tmp` directory (if it doesn't exist)
- Save each crawled page as a markdown file in the `tmp` directory
- Print a summary of successful and failed pages after completion

## Dependencies

- crawl4ai

## Configuration

You can modify the following variables in the script:
- `START_URL`: The starting URL for crawling
- `OUT_DIR`: The directory where markdown files will be saved
- Crawler settings (depth, timeouts, retry logic, etc.)

## References

- [unclecode/crawl4ai)](https://github.com/unclecode/crawl4ai)
