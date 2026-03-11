from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    case_id: str
    full_name: str
    country: str | None
    date_of_birth: str | None

    plan: dict[str, Any]
    queries: list[str]
    search_results: list[dict[str, Any]]
    articles: list[dict[str, Any]]
    findings: list[dict[str, Any]]

    summary: str
    recommendations: list[str]
    overall_score: float
    overall_risk: str

    reflection_notes: list[str]
    reflection_loop_count: int
    should_reflect_retry: bool

    guardrail_flags: list[str]
    status: str
    report: dict[str, Any]

    messages: list[dict[str, Any]]
