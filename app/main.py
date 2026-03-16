from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from app.core.settings import settings
from app.db.memory import MemoryStore
from app.models.schemas import HumanReviewRequest, ScreeningRequest, ScreeningResponse
from app.orchestrator.graph import ScreeningWorkflow
from app.orchestrator.state import GraphState
from app.tools.guardrails import check_output_guardrails, validate_name_input
from app.tools.risk import classify_many
from app.tools.scraper import scrape_many
from app.tools.search import batch_search
from app.tools.summarizer import summarize_findings
from app.tools.tool_registry import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title=settings.app_name)


def _register_tools() -> None:
    registry.register("batch_search", batch_search)
    registry.register("scrape_many", scrape_many)
    registry.register("classify_many", classify_many)
    registry.register("summarize_findings", summarize_findings)
    registry.register("check_output_guardrails", check_output_guardrails)


def _report_to_state(case_id: str, report: dict, review_action: str, modified_updates: list[dict]) -> GraphState:
    metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
    proposed_updates = metadata.get("proposed_prompt_updates", [])
    if not isinstance(proposed_updates, list):
        proposed_updates = []

    approved_updates = metadata.get("approved_prompt_updates", [])
    if not isinstance(approved_updates, list):
        approved_updates = []

    state: GraphState = {
        "case_id": case_id,
        "full_name": str(report.get("full_name", "")),
        "findings": report.get("key_findings", []),
        "summary": str(report.get("summary", "")),
        "recommendations": report.get("recommendations", []),
        "overall_score": float(report.get("overall_score", 0.0)),
        "overall_risk": str(report.get("overall_risk", "low")),
        "guardrail_flags": report.get("guardrail_flags", []),
        "reflection_notes": report.get("reflection_notes", []),
        "reflection_human_summary": str(metadata.get("reflection_human_summary", "")),
        "proposed_prompt_updates": proposed_updates,
        "approved_prompt_updates": approved_updates,
        "human_review_required": bool(metadata.get("human_review_required", True)),
        "prompt_revision_required": bool(metadata.get("prompt_revision_required", bool(proposed_updates))),
        "human_review_action": review_action,
        "human_modified_updates": modified_updates,
        "messages": [],
    }
    return state


@app.on_event("startup")
async def startup() -> None:
    _register_tools()

    memory = MemoryStore(settings.db_path)
    await memory.init()

    app.state.memory = memory
    try:
        app.state.workflow = ScreeningWorkflow(memory)
    except Exception as exc:
        logging.exception("Workflow initialization failed: %s", exc)
        app.state.workflow = None


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": settings.app_name}


@app.post("/screening/run", response_model=ScreeningResponse)
async def run_screening(request: ScreeningRequest) -> ScreeningResponse:
    input_flags = validate_name_input(request.full_name)
    if input_flags:
        raise HTTPException(status_code=400, detail={"guardrail_flags": input_flags})

    case_id = str(uuid4())
    await app.state.memory.create_case(case_id, request.full_name, status="in_progress")
    workflow = getattr(app.state, "workflow", None)
    if workflow is None:
        raise HTTPException(status_code=503, detail="Workflow unavailable")

    initial_state: GraphState = {
        "case_id": case_id,
        "full_name": request.full_name,
        "country": request.country,
        "date_of_birth": request.date_of_birth,
        "reflection_loop_count": 0,
        "human_review_action": "pending",
        "messages": [],
    }

    output_state = await workflow.run(initial_state)

    return ScreeningResponse(
        case_id=case_id,
        status=output_state.get("status", "completed"),
        report=output_state.get("report"),
    )


@app.get("/screening/{case_id}", response_model=ScreeningResponse)
async def get_screening(case_id: str) -> ScreeningResponse:
    report = await app.state.memory.get_report(case_id)
    if not report:
        raise HTTPException(status_code=404, detail="Case not found")

    status = "completed"
    metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
    if report.get("guardrail_flags") or metadata.get("human_review_required"):
        status = "needs_manual_review"

    return ScreeningResponse(case_id=case_id, status=status, report=report)


@app.post("/screening/{case_id}/review", response_model=ScreeningResponse)
async def submit_human_review(case_id: str, request: HumanReviewRequest) -> ScreeningResponse:
    workflow = getattr(app.state, "workflow", None)
    if workflow is None:
        raise HTTPException(status_code=503, detail="Workflow unavailable")

    report = await app.state.memory.get_report(case_id)
    if not report:
        raise HTTPException(status_code=404, detail="Case not found")

    if request.human_review_action == "modify_prompt_update" and not request.human_modified_updates:
        raise HTTPException(status_code=400, detail="human_modified_updates cannot be empty for modify_prompt_update")

    state = _report_to_state(
        case_id=case_id,
        report=report,
        review_action=request.human_review_action,
        modified_updates=request.human_modified_updates,
    )

    output_state = await workflow.run_human_review(state)
    logging.info(
        "HUMAN_REVIEW_SUBMITTED case_id=%s action=%s applied_prompt_stages=%s review_required=%s",
        case_id,
        request.human_review_action,
        [item.get("stage") for item in output_state.get("approved_prompt_updates", []) if isinstance(item, dict)],
        output_state.get("human_review_required", False),
    )
    return ScreeningResponse(
        case_id=case_id,
        status=output_state.get("status", "completed"),
        report=output_state.get("report"),
    )
