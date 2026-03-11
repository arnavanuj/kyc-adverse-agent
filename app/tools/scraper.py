from __future__ import annotations

import asyncio
import logging
import re

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _clean_text(text: str, max_len: int = 7000) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:max_len]


async def scrape_url(url: str, timeout_seconds: int = 15) -> dict:
    logger.info("STEP 3: Scraping article: %s", url)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; KYC-Agent/1.0; +https://example.org/bot)"
    }

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True) as response:
                html = await response.text(errors="ignore")
    except Exception as exc:
        return {"url": url, "content": "", "error": str(exc)}

    soup = BeautifulSoup(html, "html.parser")

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body_text = " ".join([p for p in paragraphs if p])

    if len(body_text) < 150:
        body_text = soup.get_text(" ", strip=True)

    cleaned = _clean_text(body_text)
    logger.info("Scraped text length: %d characters", len(cleaned))
    if len(cleaned) < 300:
        logger.warning("Article text too short, may indicate scraping failure")

    return {
        "url": url,
        "title": title,
        "content": cleaned,
    }


async def scrape_many(urls: list[str], timeout_seconds: int = 15) -> list[dict]:
    tasks = [scrape_url(url, timeout_seconds=timeout_seconds) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    parsed: list[dict] = []
    for item in results:
        if isinstance(item, Exception):
            continue
        parsed.append(item)
    return parsed
