"""Crawls a website and saves its pages as Markdown files (logging-only)."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import click
from crawler import Crawler


def _timestamped_dir(start_url: str) -> Path:
    domain = urlparse(start_url).netloc
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return Path(f"./tmp/{domain}/{stamp}")


async def _main(start_url: str):
    results = await Crawler(start_url).run()
    out_dir = _timestamped_dir(start_url)

    success = 0
    failed: list[tuple[str, str]] = []

    # Save
    for _, res in enumerate(results, 1):
        if res.success:
            success += 1
            filename = out_dir / f"{res.url.split('//')[1].replace('/', '_')}.md"
            filename.parent.mkdir(parents=True, exist_ok=True)
            filename.write_text(res.markdown or "")
        else:
            failed.append((res.url, res.error_message or "unknown error"))

    # Report summary
    print(f"Success: {success} pages")
    print(f"Failed : {len(failed)} pages")

    if failed:
        print("\n--- Skipped URLs List ---")
        for url, msg in failed:
            print(f"{url}\t{msg}")


@click.command()
@click.option("--start-url", default="https://docs.crawl4ai.com/")
def main(start_url: str) -> None:
    asyncio.run(_main(start_url))


if __name__ == "__main__":
    main()
