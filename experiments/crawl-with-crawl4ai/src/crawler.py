from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    MemoryAdaptiveDispatcher,
    RateLimiter,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import DomainFilter, FilterChain


@dataclass
class Crawler:
    start_url: str

    async def run(self) -> None:
        try:
            return await self.crawl()
        except Exception as e:
            raise e

    async def crawl(self) -> List[CrawlResult]:
        domain = urlparse(self.start_url).netloc
        filter_chain = FilterChain([DomainFilter(allowed_domains=[domain])])

        config = CrawlerRunConfig(
            remove_forms=True,
            check_robots_txt=True,
            exclude_external_links=True,
            exclude_social_media_links=True,
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=2,
                include_external=False,
                filter_chain=filter_chain,
            ),
            scraping_strategy=LXMLWebScrapingStrategy(),
            verbose=False,
        )

        dispatcher = MemoryAdaptiveDispatcher(
            rate_limiter=RateLimiter(
                base_delay=(1.0, 2.0),
                max_delay=30.0,
                max_retries=2,
            )
        )

        async with AsyncWebCrawler() as _crawler:
            return await _crawler.arun(
                self.start_url,
                config=config,
                dispatcher=dispatcher,
            )
