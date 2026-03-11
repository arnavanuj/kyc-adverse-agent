from __future__ import annotations

from app.orchestrator.state import GraphState


class ReflectionAgent:
    name = "reflection_agent"

    async def run(self, state: GraphState) -> GraphState:
        findings = state.get("findings", [])
        articles = state.get("articles", [])
        loop_count = state.get("reflection_loop_count", 0)

        notes: list[str] = []
        should_retry = False

        high_conf_count = len([f for f in findings if f.get("confidence", 0) >= 0.55])
        if len(articles) < 3:
            notes.append("Low article coverage; additional search pass recommended.")
            should_retry = True

        if findings and high_conf_count == 0:
            notes.append("Findings confidence is low; request additional corroboration.")
            should_retry = True

        if not notes:
            notes.append("Reflection passed: evidence coverage is sufficient for reporting.")

        state["reflection_notes"] = state.get("reflection_notes", []) + notes
        state["should_reflect_retry"] = should_retry and loop_count < state.get("plan", {}).get(
            "max_reflection_loops", 1
        )

        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "search_agent" if state["should_reflect_retry"] else "compliance_guardrail_agent",
                "type": "reflection_decision",
                "payload": {"retry": state["should_reflect_retry"], "notes": notes},
            }
        )
        return state
