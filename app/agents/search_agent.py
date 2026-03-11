from __future__ import annotations

from app.core.settings import settings
from app.orchestrator.state import GraphState
from app.tools.tool_registry import registry


class SearchAgent:
    name = "search_agent"

    async def run(self, state: GraphState) -> GraphState:
        queries = state.get("queries", [])
        search_tool = registry.get("batch_search")
        results = await search_tool(queries, max_results=settings.search_results_per_query)

        state["search_results"] = results[: settings.max_articles * 2]
        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "scraper_agent",
                "type": "search_results",
                "payload": {"count": len(state["search_results"])},
            }
        )
        return state
