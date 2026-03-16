import asyncio

import pytest

reflection_module = pytest.importorskip("app.agents.reflection_agent")
ReflectionAgent = reflection_module.ReflectionAgent


def _base_state():
    return {
        "case_id": "case-1",
        "full_name": "Nirav Modi",
        "articles": [{"content": "Potential sanctions violations and money laundering allegations were reported."}],
        "findings": [],
        "selected_evidence_sentences": [],
        "compressed_evidence": [],
        "overall_score": 0.5,
        "overall_risk": "medium",
        "confidence": 0.7,
        "llm_reasoning": "sanctions and money laundering concerns identified",
        "plan": {"max_reflection_loops": 1},
        "messages": [],
    }


def test_evidence_update_contains_only_incremental_severe_indicators(monkeypatch):
    def _fake_load_prompt_text(stage: str) -> str:
        if stage == "evidence_selection_prompt":
            return "Ensure severe indicators include sanctions."
        return "coverage for fraud corruption regulatory actions political exposure"

    monkeypatch.setattr(reflection_module, "load_prompt_text", _fake_load_prompt_text)

    state = _base_state()
    out = asyncio.run(ReflectionAgent().run(state))

    evidence_updates = [
        update for update in out.get("proposed_prompt_updates", []) if update.get("stage") == "evidence_selection_prompt"
    ]
    assert evidence_updates
    recommendation = evidence_updates[0]["recommendation"]
    assert "these missing indicators when present" in recommendation
    assert "Severe indicators:" in recommendation
    assert "- money laundering" in recommendation
    assert "- sanctions" not in recommendation


def test_no_evidence_update_when_no_incremental_severe_indicator(monkeypatch):
    def _fake_load_prompt_text(stage: str) -> str:
        if stage == "evidence_selection_prompt":
            return "Ensure severe indicators include sanctions and money laundering."
        return "coverage for fraud corruption regulatory actions political exposure"

    monkeypatch.setattr(reflection_module, "load_prompt_text", _fake_load_prompt_text)

    state = _base_state()
    out = asyncio.run(ReflectionAgent().run(state))

    evidence_updates = [
        update for update in out.get("proposed_prompt_updates", []) if update.get("stage") == "evidence_selection_prompt"
    ]
    assert evidence_updates == []


def test_evidence_update_includes_incremental_non_severe_indicator(monkeypatch):
    def _fake_load_prompt_text(stage: str) -> str:
        if stage == "evidence_selection_prompt":
            return "Ensure severe indicators include sanctions and money laundering."
        return "coverage for fraud corruption regulatory actions"

    monkeypatch.setattr(reflection_module, "load_prompt_text", _fake_load_prompt_text)

    state = _base_state()
    state["articles"] = [{"content": "A politically exposed person profile was reported."}]
    state["llm_reasoning"] = "General profile summary"

    out = asyncio.run(ReflectionAgent().run(state))

    evidence_updates = [
        update for update in out.get("proposed_prompt_updates", []) if update.get("stage") == "evidence_selection_prompt"
    ]
    assert evidence_updates
    recommendation = evidence_updates[0]["recommendation"]
    assert "Additional indicators:" in recommendation
    assert "- political exposure" in recommendation
