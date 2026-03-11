from __future__ import annotations


def summarize_findings(full_name: str, findings: list[dict]) -> tuple[str, list[str]]:
    risky = [f for f in findings if f.get("risk_score", 0) >= 0.35]
    labels = sorted({label for f in findings for label in f.get("risk_labels", [])})

    if not findings:
        return (
            f"No adverse media articles were retrieved for {full_name}.",
            ["Run manual review with expanded language coverage."],
        )

    if not risky:
        summary = (
            f"Screening for {full_name} found media coverage with low-risk indicators. "
            "No significant adverse media pattern was detected from collected sources."
        )
        recs = [
            "Maintain standard due diligence.",
            "Re-screen periodically or upon trigger events.",
        ]
        return summary, recs

    summary = (
        f"Screening for {full_name} identified {len(risky)} potentially adverse articles. "
        f"Detected risk themes: {', '.join(labels) if labels else 'general adverse activity'}."
    )
    recs = [
        "Escalate to enhanced due diligence (EDD).",
        "Validate identity match with additional identifiers.",
        "Review primary sources and legal records before final decision.",
    ]
    return summary, recs
