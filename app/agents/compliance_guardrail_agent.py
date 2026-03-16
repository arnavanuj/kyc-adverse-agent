from __future__ import annotations

from app.orchestrator.state import GraphState
from app.tools.tool_registry import registry


class ComplianceGuardrailAgent:
    name = "compliance_guardrail_agent"

    async def run(self, state: GraphState) -> GraphState:
        output_tool = registry.get("check_output_guardrails")
        flags = output_tool(state.get("findings", []), state.get("summary", ""))

        if not state.get("articles"):
            flags.append("No article content collected.")
        if state.get("human_review_required"):
            flags.append("Human review pending for reflection critique or prompt update.")
        if state.get("prompt_revision_required"):
            flags.append("Prompt revision recommended by reflection agent.")

        state["guardrail_flags"] = flags
        state["status"] = "needs_manual_review" if flags else "completed"
        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "report_generator_agent",
                "type": "guardrail_result",
                "payload": {"flags": flags, "status": state["status"]},
            }
        )
        return state
