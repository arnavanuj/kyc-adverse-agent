from __future__ import annotations

import logging

from app.orchestrator.state import GraphState
from app.tools.prompt_store import apply_prompt_updates

logger = logging.getLogger(__name__)


class HumanReviewAgent:
    name = "human_review_agent"

    async def run(self, state: GraphState) -> GraphState:
        action = str(state.get("human_review_action", "pending")).strip().lower()
        proposed_updates = state.get("proposed_prompt_updates", [])
        modified_updates = state.get("human_modified_updates", [])
        updates_to_apply = proposed_updates

        if action == "modify_prompt_update" and modified_updates:
            updates_to_apply = modified_updates

        applied_stages: list[str] = []
        if action == "reject_update":
            state["human_review_required"] = False
            state["prompt_revision_required"] = False
        elif action in {"approve_prompt_update", "modify_prompt_update"} and updates_to_apply:
            applied_stages = apply_prompt_updates(updates_to_apply)
            state["approved_prompt_updates"] = updates_to_apply
            state["prompt_revision_required"] = False if applied_stages else state.get("prompt_revision_required", False)
            state["human_review_required"] = False if applied_stages else True
        else:
            state["human_review_required"] = True

        summary = state.get("reflection_human_summary", "Human review required.")
        note = (
            f"Human review action: {action}. Applied prompt stages: {', '.join(applied_stages) if applied_stages else 'none'}."
        )
        state["reflection_notes"] = state.get("reflection_notes", []) + [summary, note]

        logger.info(
            "REFLECTION_FINAL_STATUS case_id=%s human_review_action=%s applied_stages=%s review_required=%s",
            state.get("case_id", "unknown"),
            action,
            applied_stages,
            state.get("human_review_required", False),
        )

        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "compliance_guardrail_agent",
                "type": "human_review_decision",
                "payload": {
                    "action": action,
                    "applied_stages": applied_stages,
                    "human_review_required": state.get("human_review_required", False),
                },
            }
        )
        return state
