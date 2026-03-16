from __future__ import annotations

from pathlib import Path

PROMPT_STORE_DIR = Path("prompt_store")

PROMPT_FILES = {
    "compression_prompt": "compression_prompt.txt",
    "reasoning_prompt": "reasoning_prompt.txt",
    "evidence_selection_prompt": "evidence_selection_prompt.txt",
}


def get_prompt_path(stage: str) -> Path | None:
    filename = PROMPT_FILES.get(stage)
    if not filename:
        return None
    return PROMPT_STORE_DIR / filename


def load_prompt_text(stage: str) -> str:
    path = get_prompt_path(stage)
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def apply_prompt_updates(updates: list[dict]) -> list[str]:
    applied: list[str] = []
    for update in updates:
        stage = str(update.get("stage", "")).strip()
        recommendation = str(update.get("recommendation", "")).strip()
        if not stage or not recommendation:
            continue

        path = get_prompt_path(stage)
        if path is None:
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        if recommendation in existing:
            continue

        separator = "\n\n" if existing and not existing.endswith("\n") else "\n"
        new_text = f"{existing}{separator}# Critic update\n{recommendation}\n".lstrip()
        path.write_text(new_text, encoding="utf-8")
        applied.append(stage)
    return applied
