"""Crawls a website and saves its content as markdown files."""

import asyncio
import pathlib
from typing import List, Tuple

from crawl4ai import (
    AsyncWebCrawler,
    CacheMode,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

START_URL = "https://docs.crawl4ai.com/"
OUT_DIR = pathlib.Path("tmp")
OUT_DIR.mkdir(exist_ok=True)


async def main() -> None:
    """Crawls the specified website and saves each page as a markdown file."""
    run_cfg = CrawlerRunConfig(
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=2, include_external=False  # Entry point + 2 levels
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        check_robots_txt=True,
        page_timeout=45_000,
        cache_mode=CacheMode.ENABLED,
        verbose=True,
    )

    dispatcher = MemoryAdaptiveDispatcher(
        rate_limiter=RateLimiter(
            base_delay=(1.0, 2.0),  # Initial wait 1-2 seconds
            max_delay=30.0,
            max_retries=2,  # Automatic retry 2 times
        )
    )

    failed: List[Tuple[str, str]] = []
    total_ok = 0

    async with AsyncWebCrawler() as crawler:
        # Important: first await and then receive
        results = await crawler.arun(START_URL, config=run_cfg, dispatcher=dispatcher)

    # `results` is List[CrawlResult] (all deep crawled pages)
    for result in results:
        if result.success:
            total_ok += 1
            filename = OUT_DIR / f"{result.url.split('//')[1].replace('/', '_')}.md"
            filename.write_text(result.markdown or "")
        else:
            failed.append((result.url, result.error_message))

    # Summary
    print(f"✅ Success: {total_ok} pages")
    print(f"❌ Failed: {len(failed)} pages")
    if failed:
        print("\n--- Skipped URLs List ---")
        for url, msg in failed:
            print(f"{url}\t{msg}")


if __name__ == "__main__":
    asyncio.run(main())
