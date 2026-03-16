from __future__ import annotations

import logging
from datetime import datetime

from app.models.schemas import ComplianceReport
from app.orchestrator.state import GraphState

logger = logging.getLogger(__name__)


class ReportGeneratorAgent:
    name = "report_generator_agent"

    async def run(self, state: GraphState) -> GraphState:
        logger.info("STEP 10: Generating final risk report")
        state["proposed_prompt_updates"] = state.get("proposed_prompt_updates", [])
        state["human_review_required"] = bool(
            state.get("human_review_required", False) or state.get("human_review_action", "pending") == "pending"
        )

        report = ComplianceReport(
            case_id=state["case_id"],
            full_name=state["full_name"],
            generated_at=datetime.utcnow(),
            overall_risk=state.get("overall_risk", "low"),
            overall_score=float(state.get("overall_score", 0.0)),
            summary=state.get("summary", ""),
            key_findings=state.get("findings", []),
            recommendations=state.get("recommendations", []),
            guardrail_flags=state.get("guardrail_flags", []),
            reflection_notes=state.get("reflection_notes", []),
            metadata={
                "article_count": len(state.get("articles", [])),
                "search_result_count": len(state.get("search_results", [])),
                "reflection_status": state.get("reflection_feedback", {}).get("reflection_status", ""),
                "reflection_confidence": state.get("reflection_confidence", 0.0),
                "prompt_revision_required": state.get("prompt_revision_required", False),
                "human_review_required": state.get("human_review_required", False),
                "reflection_human_summary": state.get("reflection_human_summary", ""),
                "proposed_prompt_updates": state.get("proposed_prompt_updates", []),
                "approved_prompt_updates": state.get("approved_prompt_updates", []),
                "human_review_action": state.get("human_review_action", "pending"),
            },
        )

        state["report"] = report.model_dump(mode="json")
        state["case_id"] = state.get("case_id", report.case_id)
        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "api",
                "type": "final_report",
                "payload": {"status": state.get("status", "completed")},
            }
        )
        if state.get("human_review_required", False):
            state.setdefault("messages", []).append(
                {
                    "from": self.name,
                    "to": "human_review_agent",
                    "type": "human_review_required",
                    "payload": {
                        "case_id": state["case_id"],
                        "proposed_updates": state.get("proposed_prompt_updates", []),
                    },
                }
            )
        return state
