"""Persistence for trips and their follow-up conversations.

Uses Postgres when DATABASE_URL (or POSTGRES_URL) is set - e.g. Neon on Vercel -
and a local SQLite file otherwise.
"""
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
# Vercel's filesystem is read-only apart from /tmp (which doesn't persist).
SQLITE_PATH = Path("/tmp" if os.getenv("VERCEL") else Path(__file__).parent) / "travelbrain.db"

if DATABASE_URL:
    import psycopg
    from psycopg.rows import dict_row


@contextmanager
def _cursor():
    if DATABASE_URL:
        with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                yield cur
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn.cursor()
        finally:
            conn.close()


def _q(sql: str) -> str:
    """Queries are written with '?' placeholders; Postgres wants '%s'."""
    return sql.replace("?", "%s") if DATABASE_URL else sql


def init() -> None:
    serial = "SERIAL" if DATABASE_URL else "INTEGER"
    autoinc = "" if DATABASE_URL else " AUTOINCREMENT"
    with _cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trips (
            id TEXT PRIMARY KEY, prompt TEXT, title TEXT, created_at TEXT,
            request_json TEXT, reports_json TEXT, plan_json TEXT
        )""")
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS messages (
            id {serial} PRIMARY KEY{autoinc}, trip_id TEXT, role TEXT,
            content TEXT, created_at TEXT
        )""")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_trip(prompt: str, request: dict, reports: dict, plan: dict) -> str:
    trip_id = uuid.uuid4().hex[:12]
    with _cursor() as cur:
        cur.execute(
            _q("INSERT INTO trips VALUES (?,?,?,?,?,?,?)"),
            (trip_id, prompt, plan.get("title", prompt[:60]), _now(),
             json.dumps(request), json.dumps(reports), json.dumps(plan)),
        )
    return trip_id


def list_trips() -> list[dict]:
    with _cursor() as cur:
        cur.execute("SELECT id, title, prompt, created_at FROM trips ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def get_trip(trip_id: str) -> dict | None:
    with _cursor() as cur:
        cur.execute(_q("SELECT * FROM trips WHERE id=?"), (trip_id,))
        row = cur.fetchone()
        if not row:
            return None
        cur.execute(_q("SELECT role, content FROM messages WHERE trip_id=? ORDER BY id"), (trip_id,))
        msgs = cur.fetchall()
    return {
        "id": row["id"], "prompt": row["prompt"], "title": row["title"], "created_at": row["created_at"],
        "request": json.loads(row["request_json"]), "reports": json.loads(row["reports_json"]),
        "plan": json.loads(row["plan_json"]), "messages": [dict(m) for m in msgs],
    }


def add_message(trip_id: str, role: str, content: str) -> None:
    with _cursor() as cur:
        cur.execute(
            _q("INSERT INTO messages (trip_id, role, content, created_at) VALUES (?,?,?,?)"),
            (trip_id, role, content, _now()),
        )


def delete_trip(trip_id: str) -> None:
    with _cursor() as cur:
        cur.execute(_q("DELETE FROM messages WHERE trip_id=?"), (trip_id,))
        cur.execute(_q("DELETE FROM trips WHERE id=?"), (trip_id,))
