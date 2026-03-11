from __future__ import annotations

from app.orchestrator.state import GraphState
from app.tools.risk import aggregate_risk
from app.tools.tool_registry import registry


class SummarizationAgent:
    name = "summarization_agent"

    async def run(self, state: GraphState) -> GraphState:
        summarize_tool = registry.get("summarize_findings")
        findings = state.get("findings", [])

        summary, recs = summarize_tool(state["full_name"], findings)
        score, level = aggregate_risk(findings)

        state["summary"] = summary
        state["recommendations"] = recs
        state["overall_score"] = score
        state["overall_risk"] = level

        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "reflection_agent",
                "type": "draft_summary",
                "payload": {"risk": level, "score": score},
            }
        )
        return state
