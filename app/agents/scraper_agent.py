from __future__ import annotations

from app.core.settings import settings
from app.orchestrator.state import GraphState
from app.tools.tool_registry import registry


class ScraperAgent:
    name = "scraper_agent"

    async def run(self, state: GraphState) -> GraphState:
        rows = state.get("search_results", [])[: settings.max_articles]
        urls = [row.get("url", "") for row in rows if row.get("url")]

        scrape_tool = registry.get("scrape_many")
        scraped = await scrape_tool(urls, timeout_seconds=settings.request_timeout_seconds)

        mapped: list[dict] = []
        by_url = {row.get("url"): row for row in rows}
        for item in scraped:
            url = item.get("url", "")
            source_row = by_url.get(url, {})
            mapped.append(
                {
                    "url": url,
                    "title": item.get("title") or source_row.get("title", ""),
                    "snippet": source_row.get("snippet", ""),
                    "source": source_row.get("source", ""),
                    "content": item.get("content", ""),
                    "published": source_row.get("published"),
                }
            )

        state["articles"] = [a for a in mapped if a.get("content")]
        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "risk_classification_agent",
                "type": "articles",
                "payload": {"count": len(state["articles"])},
            }
        )
        return state
