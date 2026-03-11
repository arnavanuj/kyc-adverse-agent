from __future__ import annotations

from app.orchestrator.state import GraphState
from app.tools.tool_registry import registry


class RiskClassificationAgent:
    name = "risk_classification_agent"

    async def run(self, state: GraphState) -> GraphState:
        articles = state.get("articles", [])
        user_query = state.get("full_name", "")
        classify_tool = registry.get("classify_many")
        findings = await classify_tool(articles, user_query=user_query)

        state["findings"] = findings
        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "summarization_agent",
                "type": "findings",
                "payload": {"count": len(findings)},
            }
        )
        return state
