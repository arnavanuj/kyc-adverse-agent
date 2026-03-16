from __future__ import annotations

import logging
from statistics import mean

from app.orchestrator.state import GraphState
from app.tools.prompt_store import load_prompt_text

logger = logging.getLogger(__name__)

REFLECTION_CONFIDENCE_THRESHOLD = 0.6

INDICATOR_KEYWORDS = {
    "fraud": ["fraud", "embezzlement", "forgery", "scam"],
    "corruption": ["corruption", "bribery", "kickback", "graft"],
    "sanctions": ["sanction", "ofac", "blacklist", "embargo"],
    "money_laundering": ["money laundering", "laundering", "aml", "hawala"],
    "criminal_investigations": ["investigation", "charged", "indicted", "arrested", "prosecuted"],
    "financial_misconduct": ["financial misconduct", "misappropriation", "insolvency", "default", "securities fraud"],
    "regulatory_actions": ["regulator", "enforcement", "penalty", "fine", "extradition", "court ordered"],
    "political_exposure": ["politically exposed", "pep", "minister", "prime minister", "public office"],
}

SEVERE_CATEGORIES = (
    "fraud",
    "corruption",
    "sanctions",
    "money_laundering",
    "criminal_investigations",
)

INDICATOR_LABELS = {
    "fraud": "fraud",
    "corruption": "corruption",
    "sanctions": "sanctions",
    "money_laundering": "money laundering",
    "criminal_investigations": "criminal investigations",
    "financial_misconduct": "financial misconduct",
    "regulatory_actions": "regulatory actions",
    "political_exposure": "political exposure",
}


def _has_keyword(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _collect_selected_evidence(findings: list[dict]) -> list[str]:
    selected: list[str] = []
    for finding in findings:
        snippets = finding.get("evidence_snippets", [])
        if isinstance(snippets, list):
            selected.extend([str(s).strip() for s in snippets if str(s).strip()])
    return selected


def _format_indicator_bullets(categories: list[str]) -> str:
    labels = [INDICATOR_LABELS.get(category, category.replace("_", " ")) for category in categories]
    return "\n".join(f"- {label}" for label in labels)


def _build_incremental_evidence_recommendation(
    severe_categories: list[str], non_severe_categories: list[str]
) -> str:
    lines = ["Before finalizing top-k evidence, enforce inclusion of sentences containing these missing indicators when present:"]
    if severe_categories:
        lines.append("Severe indicators:")
        lines.append(_format_indicator_bullets(severe_categories))
    if non_severe_categories:
        lines.append("Additional indicators:")
        lines.append(_format_indicator_bullets(non_severe_categories))
    return "\n".join(lines)


class ReflectionAgent:
    name = "reflection_agent"

    async def run(self, state: GraphState) -> GraphState:
        logger.info("REFLECTION_AGENT_START case_id=%s", state.get("case_id", "unknown"))

        findings = state.get("findings", [])
        articles = state.get("articles", [])
        loop_count = state.get("reflection_loop_count", 0)
        max_loops = state.get("plan", {}).get("max_reflection_loops", 1)

        clean_scraped_text = state.get("clean_scraped_text") or "\n\n".join(
            [str(article.get("content", "")).strip() for article in articles if article.get("content")]
        )
        selected_evidence_sentences = state.get("selected_evidence_sentences") or _collect_selected_evidence(findings)
        compressed_evidence = state.get("compressed_evidence") or selected_evidence_sentences
        risk_score = float(state.get("overall_score", 0.0))
        confidence_values = [float(item.get("confidence", 0.0)) for item in findings]
        confidence = float(state.get("confidence", mean(confidence_values) if confidence_values else 0.0))
        llm_reasoning = state.get("llm_reasoning") or " ".join(
            [str(item.get("rationale", "")).strip() for item in findings if item.get("rationale")]
        )
        classification_result = state.get("classification_result") or {
            "overall_risk": state.get("overall_risk", "low"),
            "overall_score": risk_score,
            "findings_count": len(findings),
        }

        state["clean_scraped_text"] = clean_scraped_text
        state["selected_evidence_sentences"] = selected_evidence_sentences
        state["compressed_evidence"] = compressed_evidence
        state["confidence"] = confidence
        state["llm_reasoning"] = llm_reasoning
        state["classification_result"] = classification_result

        notes: list[str] = []
        issues_detected: list[dict] = []
        prompt_improvement_suggestions: list[dict] = []
        recommended_prompt_updates: list[dict] = []
        should_retry = False

        if len(articles) < 3 and loop_count < max_loops:
            notes.append("Low article coverage; additional search pass recommended.")
            should_retry = True

        clean_text_lower = clean_scraped_text.lower()
        selected_text_lower = " ".join(selected_evidence_sentences).lower()
        compressed_text_lower = " ".join(compressed_evidence).lower()
        evidence_union_text = f"{selected_text_lower} {compressed_text_lower}".strip()

        missing_categories: list[str] = []
        for category, keywords in INDICATOR_KEYWORDS.items():
            seen_in_clean = _has_keyword(clean_text_lower, keywords)
            seen_in_evidence = _has_keyword(evidence_union_text, keywords)
            if seen_in_clean and not seen_in_evidence:
                missing_categories.append(category)
                issues_detected.append(
                    {
                        "type": "missing_evidence",
                        "description": f"Potential {category.replace('_', ' ')} signals were found in scraped text but not in selected evidence.",
                        "severity": "high",
                    }
                )

        logger.info(
            "REFLECTION_EVIDENCE_ANALYSIS missing_categories=%s selected_evidence_count=%s",
            missing_categories,
            len(selected_evidence_sentences),
        )

        severe_signal_in_clean = any(
            _has_keyword(clean_text_lower, INDICATOR_KEYWORDS[category]) for category in SEVERE_CATEGORIES
        )

        if not llm_reasoning.strip():
            issues_detected.append(
                {
                    "type": "reasoning_consistency",
                    "description": "Reasoning text is missing or empty.",
                    "severity": "medium",
                }
            )
        else:
            reasoning_lower = llm_reasoning.lower()
            overlap = any(
                _has_keyword(reasoning_lower, keywords) and _has_keyword(evidence_union_text, keywords)
                for keywords in INDICATOR_KEYWORDS.values()
            )
            if selected_evidence_sentences and not overlap:
                issues_detected.append(
                    {
                        "type": "reasoning_consistency",
                        "description": "Reasoning does not clearly reference detected evidence themes.",
                        "severity": "medium",
                    }
                )

        risk_alignment_assessment = "Risk score appears aligned with extracted evidence."
        if severe_signal_in_clean and risk_score < 0.35:
            risk_alignment_assessment = (
                "Potentially severe adverse indicators exist in source text but resulting risk_score is low."
            )
            issues_detected.append(
                {
                    "type": "risk_score_alignment",
                    "description": risk_alignment_assessment,
                    "severity": "high",
                }
            )
        elif (not severe_signal_in_clean) and risk_score >= 0.65:
            risk_alignment_assessment = (
                "High risk_score generated although severe indicators are not evident in scraped text."
            )
            issues_detected.append(
                {
                    "type": "risk_score_alignment",
                    "description": risk_alignment_assessment,
                    "severity": "medium",
                }
            )

        logger.info(
            "REFLECTION_RISK_ALIGNMENT severe_signal_in_clean=%s risk_score=%.3f confidence=%.3f",
            severe_signal_in_clean,
            risk_score,
            confidence,
        )

        stage_prompts = {
            "compression_prompt": load_prompt_text("compression_prompt"),
            "reasoning_prompt": load_prompt_text("reasoning_prompt"),
            "evidence_selection_prompt": load_prompt_text("evidence_selection_prompt"),
        }
        missing_prompt_topics: list[str] = []
        for category, keywords in INDICATOR_KEYWORDS.items():
            if any(_has_keyword(prompt_text, keywords) for prompt_text in stage_prompts.values()):
                continue
            missing_prompt_topics.append(category)

        if missing_prompt_topics:
            recommendation = (
                "Add explicit adverse-media indicator coverage for: "
                + ", ".join(topic.replace("_", " ") for topic in missing_prompt_topics)
                + "."
            )
            prompt_improvement_suggestions.append(
                {
                    "stage": "compression_prompt",
                    "recommendation": recommendation,
                    "reason": "Prompt coverage misses required adverse media indicators.",
                }
            )
            recommended_prompt_updates.append(
                {
                    "stage": "compression_prompt",
                    "recommendation": (
                        "Prioritize extraction of financial crime and enforcement terms including fraud, corruption, "
                        "sanctions violations, money laundering, criminal investigation signals, and regulatory actions."
                    ),
                    "reason": "Improve risk indicator coverage during compression.",
                }
            )

        if missing_categories:
            evidence_prompt_text = stage_prompts.get("evidence_selection_prompt", "").lower()
            incremental_missing_categories = [
                category
                for category in missing_categories
                if not _has_keyword(evidence_prompt_text, INDICATOR_KEYWORDS[category])
            ]
            incremental_missing_severe_categories = [
                category for category in incremental_missing_categories if category in SEVERE_CATEGORIES
            ]
            incremental_missing_non_severe_categories = [
                category for category in incremental_missing_categories if category not in SEVERE_CATEGORIES
            ]
            prompt_improvement_suggestions.append(
                {
                    "stage": "evidence_selection_prompt",
                    "recommendation": (
                        "Ensure sentence selection captures all missing allegation statements before selecting top evidence."
                    ),
                    "reason": "Signals were present in scraped text but absent from selected evidence.",
                }
            )
            if incremental_missing_categories:
                recommended_prompt_updates.append(
                    {
                        "stage": "evidence_selection_prompt",
                        "recommendation": _build_incremental_evidence_recommendation(
                            incremental_missing_severe_categories,
                            incremental_missing_non_severe_categories,
                        ),
                        "reason": "Add only newly missing indicators to avoid duplicate critic updates.",
                    }
                )

        if any(issue["type"] == "reasoning_consistency" for issue in issues_detected):
            prompt_improvement_suggestions.append(
                {
                    "stage": "reasoning_prompt",
                    "recommendation": (
                        "Require line-of-evidence traceability: reasoning must cite concrete extracted evidence and explain score mapping."
                    ),
                    "reason": "Reasoning consistency checks failed.",
                }
            )
            recommended_prompt_updates.append(
                {
                    "stage": "reasoning_prompt",
                    "recommendation": (
                        "Explain why each risk label is assigned using explicit evidence snippets and justify final risk_score calibration."
                    ),
                    "reason": "Improve reasoning consistency and auditability.",
                }
            )

        logger.info(
            "REFLECTION_PROMPT_SUGGESTIONS count=%s missing_prompt_topics=%s",
            len(prompt_improvement_suggestions),
            missing_prompt_topics,
        )

        high_severity_count = len([issue for issue in issues_detected if issue.get("severity") == "high"])
        medium_severity_count = len([issue for issue in issues_detected if issue.get("severity") == "medium"])
        reflection_confidence = max(
            0.0,
            min(
                1.0,
                0.92
                - (0.14 * high_severity_count)
                - (0.08 * medium_severity_count)
                - (0.08 if not selected_evidence_sentences else 0.0)
                - (0.08 if confidence < 0.35 else 0.0),
            ),
        )

        reflection_status = "PASS"
        if issues_detected:
            reflection_status = "REVIEW_REQUIRED"
        if reflection_confidence < REFLECTION_CONFIDENCE_THRESHOLD:
            reflection_status = "HUMAN_REVIEW_REQUIRED"

        confidence_assessment = (
            "Reflection confidence is limited due to evidence sparsity or conflicting signals."
            if reflection_status == "HUMAN_REVIEW_REQUIRED"
            else "Reflection confidence is sufficient for automated workflow routing."
        )
        prompt_revision_required = reflection_status == "REVIEW_REQUIRED" or bool(prompt_improvement_suggestions)
        human_review_required = reflection_status == "HUMAN_REVIEW_REQUIRED"

        human_summary = (
            "Reflection Agent detected coverage/consistency concerns. "
            + ("Missing categories: " + ", ".join(missing_categories) + ". " if missing_categories else "")
            + (
                "Prompt updates are suggested before applying automatic changes."
                if prompt_revision_required
                else "No prompt update required."
            )
        )

        if reflection_status == "PASS":
            notes.append("Reflection passed: evidence coverage and risk alignment are acceptable.")
        else:
            notes.append(human_summary)

        reflection_feedback = {
            "reflection_status": reflection_status,
            "issues_detected": issues_detected,
            "prompt_improvement_suggestions": prompt_improvement_suggestions,
            "risk_alignment_assessment": risk_alignment_assessment,
            "confidence_assessment": confidence_assessment,
            "recommended_prompt_updates": recommended_prompt_updates,
            "reflection_confidence": round(reflection_confidence, 3),
            "human_readable_summary": human_summary,
        }

        state["reflection_notes"] = state.get("reflection_notes", []) + notes
        state["should_reflect_retry"] = should_retry and loop_count < max_loops
        state["reflection_feedback"] = reflection_feedback
        state["reflection_confidence"] = round(reflection_confidence, 3)
        state["reflection_human_summary"] = human_summary
        state["prompt_revision_required"] = prompt_revision_required
        state["proposed_prompt_updates"] = recommended_prompt_updates
        state["human_review_required"] = human_review_required

        state.setdefault("messages", []).append(
            {
                "from": self.name,
                "to": "human_review_agent"
                if human_review_required
                else ("search_agent" if state["should_reflect_retry"] else "compliance_guardrail_agent"),
                "type": "reflection_decision",
                "payload": {
                    "retry": state["should_reflect_retry"],
                    "notes": notes,
                    "reflection_status": reflection_status,
                    "prompt_revision_required": prompt_revision_required,
                    "human_review_required": human_review_required,
                    "reflection_confidence": round(reflection_confidence, 3),
                },
            }
        )

        logger.info(
            "REFLECTION_FINAL_STATUS status=%s prompt_revision_required=%s human_review_required=%s confidence=%.3f",
            reflection_status,
            prompt_revision_required,
            human_review_required,
            reflection_confidence,
        )
        return state
