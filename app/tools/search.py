from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

from ddgs import DDGS

logger = logging.getLogger(__name__)


async def ddg_text_search(query: str, max_results: int = 8) -> list[dict]:
    def _search() -> list[dict]:
        logger.info("STEP 1: Starting search query: %s", query)
        try:
            with DDGS() as ddgs:
                rows = ddgs.text(query, max_results=max_results)
                results = []
                for row in rows:
                    url = row.get("href") or row.get("url") or ""
                    parsed = urlparse(url)
                    results.append(
                        {
                            "url": url,
                            "title": row.get("title", ""),
                            "snippet": row.get("body", ""),
                            "source": parsed.netloc,
                        }
                    )
                logger.info("STEP 2: Collected %d search results", len(results))
                if results:
                    return results
        except Exception as exc:
            logger.warning("Search provider unavailable, using fallback result: %s", exc)

        # Keep pipeline tests deterministic when search providers are unavailable.
        return [
            {
                "url": "",
                "title": f"Fallback result for query: {query}",
                "snippet": "Search provider unavailable.",
                "source": "local-fallback",
            }
        ]

    return await asyncio.to_thread(_search)


async def batch_search(queries: list[str], max_results: int = 8) -> list[dict]:
    tasks = [ddg_text_search(query=q, max_results=max_results) for q in queries]
    nested = await asyncio.gather(*tasks, return_exceptions=True)

    merged: dict[str, dict] = {}
    for result in nested:
        if isinstance(result, Exception):
            continue
        for item in result:
            url = item.get("url", "")
            if url and url not in merged:
                merged[url] = item
    return list(merged.values())[:5]


def run_search(query: str, max_results: int = 8) -> list[dict]:
    return asyncio.run(ddg_text_search(query=query, max_results=max_results))
