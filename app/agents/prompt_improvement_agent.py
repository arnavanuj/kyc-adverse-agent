from __future__ import annotations

import logging

from app.orchestrator.state import GraphState

logger = logging.getLogger(__name__)


class PromptImprovementAgent:
    name = "prompt_improvement_agent"

    async def run(self, state: GraphState) -> GraphState:
        feedback = state.get("reflection_feedback", {})
        suggestions = feedback.get("recommended_prompt_updates", []) if isinstance(feedback, dict) else []
        if not suggestions:
            suggestions = state.get("proposed_prompt_updates", [])

        state["proposed_prompt_updates"] = suggestions
        state["prompt_revision_required"] = bool(suggestions)

        logger.info(
            "REFLECTION_PROMPT_SUGGESTIONS prepared_for_human_review=%s suggestions=%s",
            state.get("prompt_revision_required", False),
            len(suggestions),
        )

        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "human_review_agent",
                "type": "prompt_improvement_ready",
                "payload": {
                    "count": len(suggestions),
                    "required": state.get("prompt_revision_required", False),
                },
            }
        )
        return state
