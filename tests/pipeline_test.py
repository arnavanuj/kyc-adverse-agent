import pytest

search_module = pytest.importorskip("app.tools.search")
risk_module = pytest.importorskip("app.tools.risk")

run_search = search_module.run_search
select_top_semantic_chunks = risk_module.select_top_semantic_chunks


def test_pipeline() -> None:
    results = run_search("Harshad Mehta scam")
    assert isinstance(results, list)


def test_semantic_selector() -> None:
    chunks = [
        "This is a sports article about football.",
        "Harshad Mehta was linked to a securities scam investigation.",
        "This paragraph is about cooking recipes.",
    ]
    selected, scores = select_top_semantic_chunks(chunks, "Harshad Mehta scam")
    assert len(selected) > 0
    assert len(scores) > 0
