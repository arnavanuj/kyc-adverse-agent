import pytest

risk = pytest.importorskip("app.tools.risk")


def test_compression_prompt_uses_prompt_store_guidance(monkeypatch):
    monkeypatch.setattr(
        risk,
        "load_prompt_text",
        lambda stage: "Use compliance-specific extraction guidance." if stage == "compression_prompt" else "",
    )

    prompt = risk._build_compression_prompt(
        "Jane Doe",
        {"sentences": ["She was investigated for sanctions evasion."]},
    )

    assert "Use compliance-specific extraction guidance." in prompt
    assert "She was investigated for sanctions evasion." in prompt


def test_reasoning_prompt_uses_prompt_store_guidance(monkeypatch):
    monkeypatch.setattr(
        risk,
        "load_prompt_text",
        lambda stage: "Justify risk score with explicit evidence links." if stage == "reasoning_prompt" else "",
    )

    prompt = risk._build_classification_prompt("EVIDENCE 1\nFact: sample")

    assert "Justify risk score with explicit evidence links." in prompt
    assert "Return STRICT JSON with the following fields" in prompt


def test_evidence_selection_query_uses_prompt_store_guidance(monkeypatch):
    monkeypatch.setattr(
        risk,
        "load_prompt_text",
        lambda stage: "Include severe fraud and corruption indicators first." if stage == "evidence_selection_prompt" else "",
    )

    query = risk._build_evidence_selection_query("Jane Doe")

    assert "Jane Doe" in query
    assert "Include severe fraud and corruption indicators first." in query
