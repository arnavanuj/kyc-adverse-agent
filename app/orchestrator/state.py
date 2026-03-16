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
    reflection_feedback: dict[str, Any]
    prompt_revision_required: bool
    human_review_required: bool
    reflection_confidence: float
    reflection_human_summary: str

    clean_scraped_text: str
    selected_evidence_sentences: list[str]
    compressed_evidence: list[str]
    llm_reasoning: str
    classification_result: dict[str, Any]
    confidence: float

    proposed_prompt_updates: list[dict[str, Any]]
    approved_prompt_updates: list[dict[str, Any]]
    human_review_action: str
    human_modified_updates: list[dict[str, Any]]

    guardrail_flags: list[str]
    status: str
    report: dict[str, Any]

    messages: list[dict[str, Any]]
