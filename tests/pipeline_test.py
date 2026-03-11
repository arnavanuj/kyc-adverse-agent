from app.tools.search import run_search
from app.tools.risk import select_top_semantic_chunks


def test_pipeline() -> None:
    results = run_search("Harshad Mehta scam")
    assert len(results) > 0


def test_semantic_selector() -> None:
    chunks = [
        "This is a sports article about football.",
        "Harshad Mehta was linked to a securities scam investigation.",
        "This paragraph is about cooking recipes.",
    ]
    selected, scores = select_top_semantic_chunks(chunks, "Harshad Mehta scam")
    assert len(selected) > 0
    assert len(scores) > 0
