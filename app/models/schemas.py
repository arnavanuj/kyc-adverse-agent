from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ScreeningRequest(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    country: str | None = None
    date_of_birth: str | None = None


class Article(BaseModel):
    url: str
    title: str
    snippet: str = ""
    source: str = ""
    published: str | None = None
    content: str = ""


class RiskFinding(BaseModel):
    url: str
    title: str
    risk_labels: list[str] = Field(default_factory=list)
    risk_score: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    evidence_snippets: list[str] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    case_id: str
    full_name: str
    generated_at: datetime
    overall_risk: str
    overall_score: float
    summary: str
    key_findings: list[RiskFinding]
    recommendations: list[str]
    guardrail_flags: list[str] = Field(default_factory=list)
    reflection_notes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScreeningResponse(BaseModel):
    case_id: str
    status: str
    report: ComplianceReport | None = None


class PersistedCase(BaseModel):
    case_id: str
    full_name: str
    status: str
    created_at: datetime
