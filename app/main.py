from __future__ import annotations

import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from app.core.settings import settings
from app.db.memory import MemoryStore
from app.models.schemas import ScreeningRequest, ScreeningResponse
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


@app.on_event("startup")
async def startup() -> None:
    _register_tools()

    memory = MemoryStore(settings.db_path)
    await memory.init()

    app.state.memory = memory
    app.state.workflow = ScreeningWorkflow(memory)


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

    initial_state: GraphState = {
        "case_id": case_id,
        "full_name": request.full_name,
        "country": request.country,
        "date_of_birth": request.date_of_birth,
        "reflection_loop_count": 0,
        "messages": [],
    }

    output_state = await app.state.workflow.run(initial_state)

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
    if report.get("guardrail_flags"):
        status = "needs_manual_review"

    return ScreeningResponse(case_id=case_id, status=status, report=report)
