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
            },
        )

        state["report"] = report.model_dump(mode="json")
        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "api",
                "type": "final_report",
                "payload": {"status": state.get("status", "completed")},
            }
        )
        return state
