import json
from datetime import datetime

import aiosqlite


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path

    async def init(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    snippet TEXT,
                    source TEXT,
                    published TEXT,
                    content TEXT
                );

                CREATE TABLE IF NOT EXISTS findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    risk_labels TEXT,
                    risk_score REAL,
                    confidence REAL,
                    rationale TEXT,
                    evidence_snippets TEXT
                );

                CREATE TABLE IF NOT EXISTS reports (
                    case_id TEXT PRIMARY KEY,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            await db.commit()

    async def create_case(self, case_id: str, full_name: str, status: str = "started") -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO cases (case_id, full_name, status, created_at) VALUES (?, ?, ?, ?)",
                (case_id, full_name, status, datetime.utcnow().isoformat()),
            )
            await db.commit()

    async def update_case_status(self, case_id: str, status: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("UPDATE cases SET status = ? WHERE case_id = ?", (status, case_id))
            await db.commit()

    async def add_message(self, case_id: str, agent_name: str, payload: dict) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO messages (case_id, agent_name, payload, created_at) VALUES (?, ?, ?, ?)",
                (case_id, agent_name, json.dumps(payload), datetime.utcnow().isoformat()),
            )
            await db.commit()

    async def save_sources(self, case_id: str, sources: list[dict]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            for src in sources:
                await db.execute(
                    """
                    INSERT INTO sources (case_id, url, title, snippet, source, published, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_id,
                        src.get("url", ""),
                        src.get("title", ""),
                        src.get("snippet", ""),
                        src.get("source", ""),
                        src.get("published"),
                        src.get("content", ""),
                    ),
                )
            await db.commit()

    async def save_findings(self, case_id: str, findings: list[dict]) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            for finding in findings:
                await db.execute(
                    """
                    INSERT INTO findings (case_id, url, title, risk_labels, risk_score, confidence, rationale, evidence_snippets)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        case_id,
                        finding.get("url", ""),
                        finding.get("title", ""),
                        json.dumps(finding.get("risk_labels", [])),
                        float(finding.get("risk_score", 0.0)),
                        float(finding.get("confidence", 0.0)),
                        finding.get("rationale", ""),
                        json.dumps(finding.get("evidence_snippets", [])),
                    ),
                )
            await db.commit()

    async def save_report(self, case_id: str, report: dict) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO reports (case_id, report_json, created_at) VALUES (?, ?, ?)",
                (case_id, json.dumps(report), datetime.utcnow().isoformat()),
            )
            await db.commit()

    async def get_report(self, case_id: str) -> dict | None:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT report_json FROM reports WHERE case_id = ?", (case_id,))
            row = await cursor.fetchone()
            if not row:
                return None
            return json.loads(row[0])
