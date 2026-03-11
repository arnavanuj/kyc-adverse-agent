from __future__ import annotations

import re


def validate_name_input(full_name: str) -> list[str]:
    flags: list[str] = []
    if len(full_name.strip()) < 2:
        flags.append("Name too short.")
    if re.search(r"[^a-zA-Z\s\-\.']", full_name):
        flags.append("Name contains unsupported characters.")
    return flags


def check_output_guardrails(findings: list[dict], summary: str) -> list[str]:
    flags: list[str] = []

    if findings and all(not item.get("evidence_snippets") for item in findings if item.get("risk_score", 0) > 0.3):
        flags.append("Risk findings lack evidence snippets.")

    if len(summary) < 40:
        flags.append("Summary is too short.")

    high_risk_without_conf = [
        f for f in findings if f.get("risk_score", 0) >= 0.65 and f.get("confidence", 0) < 0.5
    ]
    if high_risk_without_conf:
        flags.append("High risk detected with low confidence; requires manual review.")

    return flags
