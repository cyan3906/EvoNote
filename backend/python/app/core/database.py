import sqlite3
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.core.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = Path(settings.sqlite_database_path)

if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = PROJECT_ROOT / DATABASE_PATH


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS note_representations (
                note_id TEXT PRIMARY KEY,
                l2_summary TEXT NOT NULL DEFAULT '',
                l3_text TEXT NOT NULL DEFAULT '',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                vector_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL,
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS note_blocks (
                id TEXT PRIMARY KEY,
                note_id TEXT NOT NULL,
                block_index INTEGER NOT NULL,
                heading TEXT NOT NULL DEFAULT '',
                l1_text TEXT NOT NULL DEFAULT '',
                l2_summary TEXT NOT NULL DEFAULT '',
                l3_text TEXT NOT NULL DEFAULT '',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_claims (
                id TEXT PRIMARY KEY,
                note_id TEXT NOT NULL,
                block_id TEXT NOT NULL,
                claim_index INTEGER NOT NULL,
                claim_text TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                predicate TEXT NOT NULL DEFAULT '',
                object_text TEXT NOT NULL DEFAULT '',
                source_text TEXT NOT NULL DEFAULT '',
                keywords_json TEXT NOT NULL DEFAULT '[]',
                vector_json TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 0.7,
                created_at TEXT NOT NULL,
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE,
                FOREIGN KEY (block_id) REFERENCES note_blocks(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS merge_suggestions (
                id TEXT PRIMARY KEY,
                source_note_id TEXT NOT NULL,
                target_note_id TEXT NOT NULL,
                source_claim_id TEXT,
                target_claim_id TEXT,
                relation TEXT NOT NULL,
                confidence REAL NOT NULL,
                risk_level TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                patch_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_note_id) REFERENCES notes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_note_id) REFERENCES notes(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS note_versions (
                id TEXT PRIMARY KEY,
                note_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evolution_scan_jobs (
                id TEXT PRIMARY KEY,
                note_id TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT '',
                finished_at TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (note_id) REFERENCES notes(id) ON DELETE CASCADE
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_evolution_scan_jobs_note_status ON evolution_scan_jobs(note_id, status)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_evolution_scan_jobs_status_updated ON evolution_scan_jobs(status, updated_at)")


def row_to_note(row: sqlite3.Row) -> dict[str, str]:
    return {
        "id": row["id"],
        "title": row["title"],
        "tags": row["tags"],
        "body": row["body"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_notes(query: str | None = None) -> list[dict[str, str]]:
    init_db()

    with connect() as connection:
        if query:
            keyword = f"%{query}%"
            rows = connection.execute(
                """
                SELECT id, title, tags, body, created_at, updated_at
                FROM notes
                WHERE title LIKE ? OR tags LIKE ? OR body LIKE ?
                ORDER BY updated_at DESC
                """,
                (keyword, keyword, keyword),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, title, tags, body, created_at, updated_at
                FROM notes
                ORDER BY updated_at DESC
                """
            ).fetchall()

    return [row_to_note(row) for row in rows]


def get_note(note_id: str) -> dict[str, str] | None:
    init_db()

    with connect() as connection:
        row = connection.execute(
            """
            SELECT id, title, tags, body, created_at, updated_at
            FROM notes
            WHERE id = ?
            """,
            (note_id,),
        ).fetchone()

    return row_to_note(row) if row else None


def create_note(title: str = "", tags: str = "", body: str = "") -> dict[str, str]:
    init_db()
    created_at = now_iso()
    note = {
        "id": str(uuid4()),
        "title": title,
        "tags": tags,
        "body": body,
        "created_at": created_at,
        "updated_at": created_at,
    }

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO notes (id, title, tags, body, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                note["id"],
                note["title"],
                note["tags"],
                note["body"],
                note["created_at"],
                note["updated_at"],
            ),
        )

    return note


def update_note(note_id: str, title: str, tags: str, body: str) -> dict[str, str] | None:
    init_db()
    updated_at = now_iso()
    had_analysis = False

    with connect() as connection:
        existing = connection.execute(
            """
            SELECT id, title, tags, body, created_at, updated_at
            FROM notes
            WHERE id = ?
            """,
            (note_id,),
        ).fetchone()

        if existing is None:
            return None

        if existing["title"] == title and existing["tags"] == tags and existing["body"] == body:
            return row_to_note(existing)

        had_analysis = (
            connection.execute(
                "SELECT 1 FROM note_representations WHERE note_id = ? LIMIT 1",
                (note_id,),
            ).fetchone()
            is not None
        )
        cursor = connection.execute(
            """
            UPDATE notes
            SET title = ?, tags = ?, body = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, tags, body, updated_at, note_id),
        )

    if cursor.rowcount == 0:
        return None

    if had_analysis:
        _delete_external_note_indexes(note_id)

    return get_note(note_id)


def delete_note(note_id: str) -> bool:
    init_db()
    had_analysis = False

    with connect() as connection:
        had_analysis = (
            connection.execute(
                "SELECT 1 FROM note_representations WHERE note_id = ? LIMIT 1",
                (note_id,),
            ).fetchone()
            is not None
        )
        connection.execute("DELETE FROM merge_suggestions WHERE source_note_id = ? OR target_note_id = ?", (note_id, note_id))
        connection.execute("DELETE FROM knowledge_claims WHERE note_id = ?", (note_id,))
        connection.execute("DELETE FROM note_blocks WHERE note_id = ?", (note_id,))
        connection.execute("DELETE FROM note_representations WHERE note_id = ?", (note_id,))
        connection.execute("DELETE FROM note_versions WHERE note_id = ?", (note_id,))
        cursor = connection.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    if cursor.rowcount > 0 and had_analysis:
        _delete_external_note_indexes(note_id)

    return cursor.rowcount > 0


def _delete_external_note_indexes(note_id: str) -> None:
    try:
        from app.core.retrieval import delete_note_indexes

        delete_note_indexes(note_id)
    except Exception:
        return


def row_to_representation(row: sqlite3.Row) -> dict[str, object]:
    return {
        "note_id": row["note_id"],
        "l2_summary": row["l2_summary"],
        "l3_text": row["l3_text"],
        "keywords": json.loads(row["keywords_json"]),
        "vector": json.loads(row["vector_json"]),
        "updated_at": row["updated_at"],
    }


def row_to_block(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "note_id": row["note_id"],
        "block_index": row["block_index"],
        "heading": row["heading"],
        "l1_text": row["l1_text"],
        "l2_summary": row["l2_summary"],
        "l3_text": row["l3_text"],
        "keywords": json.loads(row["keywords_json"]),
        "created_at": row["created_at"],
    }


def row_to_claim(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "note_id": row["note_id"],
        "block_id": row["block_id"],
        "claim_index": row["claim_index"],
        "claim_text": row["claim_text"],
        "subject": row["subject"],
        "predicate": row["predicate"],
        "object_text": row["object_text"],
        "source_text": row["source_text"],
        "keywords": json.loads(row["keywords_json"]),
        "vector": json.loads(row["vector_json"]),
        "confidence": row["confidence"],
        "created_at": row["created_at"],
    }


def row_to_suggestion(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "source_note_id": row["source_note_id"],
        "target_note_id": row["target_note_id"],
        "source_claim_id": row["source_claim_id"],
        "target_claim_id": row["target_claim_id"],
        "relation": row["relation"],
        "confidence": row["confidence"],
        "risk_level": row["risk_level"],
        "reason": row["reason"],
        "patch": json.loads(row["patch_json"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def row_to_scan_job(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "note_id": row["note_id"],
        "status": row["status"],
        "error": row["error"],
        "attempts": row["attempts"],
        "max_attempts": row["max_attempts"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


def replace_note_analysis(
    note_id: str,
    representation: dict[str, object],
    blocks: list[dict[str, object]],
    claims: list[dict[str, object]],
) -> None:
    init_db()
    updated_at = now_iso()

    with connect() as connection:
        connection.execute("DELETE FROM knowledge_claims WHERE note_id = ?", (note_id,))
        connection.execute("DELETE FROM note_blocks WHERE note_id = ?", (note_id,))
        connection.execute(
            """
            INSERT INTO note_representations (note_id, l2_summary, l3_text, keywords_json, vector_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(note_id) DO UPDATE SET
                l2_summary = excluded.l2_summary,
                l3_text = excluded.l3_text,
                keywords_json = excluded.keywords_json,
                vector_json = excluded.vector_json,
                updated_at = excluded.updated_at
            """,
            (
                note_id,
                str(representation.get("l2_summary", "")),
                str(representation.get("l3_text", "")),
                json.dumps(representation.get("keywords", []), ensure_ascii=False),
                json.dumps(representation.get("vector", []), ensure_ascii=False),
                updated_at,
            ),
        )

        for block in blocks:
            connection.execute(
                """
                INSERT INTO note_blocks (id, note_id, block_index, heading, l1_text, l2_summary, l3_text, keywords_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block["id"],
                    note_id,
                    block["block_index"],
                    block["heading"],
                    block["l1_text"],
                    block["l2_summary"],
                    block["l3_text"],
                    json.dumps(block.get("keywords", []), ensure_ascii=False),
                    updated_at,
                ),
            )

        for claim in claims:
            connection.execute(
                """
                INSERT INTO knowledge_claims (
                    id, note_id, block_id, claim_index, claim_text, subject, predicate,
                    object_text, source_text, keywords_json, vector_json, confidence, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim["id"],
                    note_id,
                    claim["block_id"],
                    claim["claim_index"],
                    claim["claim_text"],
                    claim["subject"],
                    claim["predicate"],
                    claim["object_text"],
                    claim["source_text"],
                    json.dumps(claim.get("keywords", []), ensure_ascii=False),
                    json.dumps(claim.get("vector", []), ensure_ascii=False),
                    claim.get("confidence", 0.7),
                    updated_at,
                ),
            )


def list_claims(note_id: str | None = None, exclude_note_id: str | None = None) -> list[dict[str, object]]:
    init_db()
    clauses = []
    values: list[str] = []

    if note_id:
        clauses.append("note_id = ?")
        values.append(note_id)

    if exclude_note_id:
        clauses.append("note_id != ?")
        values.append(exclude_note_id)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with connect() as connection:
        rows = connection.execute(
            f"""
            SELECT id, note_id, block_id, claim_index, claim_text, subject, predicate,
                   object_text, source_text, keywords_json, vector_json, confidence, created_at
            FROM knowledge_claims
            {where_sql}
            ORDER BY created_at DESC, claim_index ASC
            """,
            tuple(values),
        ).fetchall()

    return [row_to_claim(row) for row in rows]


def get_note_representation(note_id: str) -> dict[str, object] | None:
    init_db()

    with connect() as connection:
        row = connection.execute(
            """
            SELECT note_id, l2_summary, l3_text, keywords_json, vector_json, updated_at
            FROM note_representations
            WHERE note_id = ?
            """,
            (note_id,),
        ).fetchone()

    return row_to_representation(row) if row else None


def list_note_blocks(note_id: str) -> list[dict[str, object]]:
    init_db()

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, note_id, block_index, heading, l1_text, l2_summary, l3_text, keywords_json, created_at
            FROM note_blocks
            WHERE note_id = ?
            ORDER BY block_index ASC
            """,
            (note_id,),
        ).fetchall()

    return [row_to_block(row) for row in rows]


def create_merge_suggestions(suggestions: list[dict[str, object]]) -> list[dict[str, object]]:
    if not suggestions:
        return []

    init_db()
    created_at = now_iso()
    source_note_ids = sorted({str(suggestion["source_note_id"]) for suggestion in suggestions})

    with connect() as connection:
        for source_note_id in source_note_ids:
            connection.execute(
                "UPDATE merge_suggestions SET status = 'superseded', updated_at = ? WHERE source_note_id = ? AND status = 'pending'",
                (created_at, source_note_id),
            )

        for suggestion in suggestions:
            connection.execute(
                """
                INSERT INTO merge_suggestions (
                    id, source_note_id, target_note_id, source_claim_id, target_claim_id,
                    relation, confidence, risk_level, reason, patch_json, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    suggestion["id"],
                    suggestion["source_note_id"],
                    suggestion["target_note_id"],
                    suggestion.get("source_claim_id"),
                    suggestion.get("target_claim_id"),
                    suggestion["relation"],
                    suggestion["confidence"],
                    suggestion["risk_level"],
                    suggestion["reason"],
                    json.dumps(suggestion["patch"], ensure_ascii=False),
                    created_at,
                    created_at,
                ),
            )

    return list_merge_suggestions(source_note_id=source_note_ids[0], status="pending")


def supersede_pending_suggestions(source_note_id: str) -> None:
    init_db()
    updated_at = now_iso()

    with connect() as connection:
        connection.execute(
            "UPDATE merge_suggestions SET status = 'superseded', updated_at = ? WHERE source_note_id = ? AND status = 'pending'",
            (updated_at, source_note_id),
        )


def list_merge_suggestions(
    source_note_id: str | None = None,
    target_note_id: str | None = None,
    status: str | None = "pending",
) -> list[dict[str, object]]:
    init_db()
    clauses = []
    values: list[str] = []

    if source_note_id:
        clauses.append("source_note_id = ?")
        values.append(source_note_id)

    if target_note_id:
        clauses.append("target_note_id = ?")
        values.append(target_note_id)

    if status:
        clauses.append("status = ?")
        values.append(status)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with connect() as connection:
        rows = connection.execute(
            f"""
            SELECT id, source_note_id, target_note_id, source_claim_id, target_claim_id,
                   relation, confidence, risk_level, reason, patch_json, status, created_at, updated_at
            FROM merge_suggestions
            {where_sql}
            ORDER BY confidence DESC, created_at DESC
            """,
            tuple(values),
        ).fetchall()

    return [row_to_suggestion(row) for row in rows]


def get_merge_suggestion(suggestion_id: str) -> dict[str, object] | None:
    init_db()

    with connect() as connection:
        row = connection.execute(
            """
            SELECT id, source_note_id, target_note_id, source_claim_id, target_claim_id,
                   relation, confidence, risk_level, reason, patch_json, status, created_at, updated_at
            FROM merge_suggestions
            WHERE id = ?
            """,
            (suggestion_id,),
        ).fetchone()

    return row_to_suggestion(row) if row else None


def update_merge_suggestion_status(suggestion_id: str, status: str) -> dict[str, object] | None:
    init_db()
    updated_at = now_iso()

    with connect() as connection:
        cursor = connection.execute(
            "UPDATE merge_suggestions SET status = ?, updated_at = ? WHERE id = ?",
            (status, updated_at, suggestion_id),
        )

    if cursor.rowcount == 0:
        return None

    return get_merge_suggestion(suggestion_id)


def save_note_version(note_id: str, reason: str = "") -> dict[str, str] | None:
    note = get_note(note_id)

    if not note:
        return None

    created_at = now_iso()
    version = {
        "id": str(uuid4()),
        "note_id": note_id,
        "title": note["title"],
        "tags": note["tags"],
        "body": note["body"],
        "reason": reason,
        "created_at": created_at,
    }

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO note_versions (id, note_id, title, tags, body, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version["id"],
                version["note_id"],
                version["title"],
                version["tags"],
                version["body"],
                version["reason"],
                version["created_at"],
            ),
        )

    return version


def create_scan_job(note_id: str, *, max_attempts: int = 3) -> dict[str, object]:
    init_db()
    created_at = now_iso()
    job = {
        "id": str(uuid4()),
        "note_id": note_id,
        "status": "queued",
        "error": "",
        "attempts": 0,
        "max_attempts": max(1, int(max_attempts)),
        "created_at": created_at,
        "updated_at": created_at,
        "started_at": "",
        "finished_at": "",
    }

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO evolution_scan_jobs (
                id, note_id, status, error, attempts, max_attempts,
                created_at, updated_at, started_at, finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job["id"],
                job["note_id"],
                job["status"],
                job["error"],
                job["attempts"],
                job["max_attempts"],
                job["created_at"],
                job["updated_at"],
                job["started_at"],
                job["finished_at"],
            ),
        )

    return job


def get_scan_job(job_id: str) -> dict[str, object] | None:
    init_db()

    with connect() as connection:
        row = connection.execute(
            """
            SELECT id, note_id, status, error, attempts, max_attempts,
                   created_at, updated_at, started_at, finished_at
            FROM evolution_scan_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

    return row_to_scan_job(row) if row else None


def get_latest_scan_job(note_id: str) -> dict[str, object] | None:
    init_db()

    with connect() as connection:
        row = connection.execute(
            """
            SELECT id, note_id, status, error, attempts, max_attempts,
                   created_at, updated_at, started_at, finished_at
            FROM evolution_scan_jobs
            WHERE note_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (note_id,),
        ).fetchone()

    return row_to_scan_job(row) if row else None


def get_active_scan_job(note_id: str) -> dict[str, object] | None:
    init_db()

    with connect() as connection:
        row = connection.execute(
            """
            SELECT id, note_id, status, error, attempts, max_attempts,
                   created_at, updated_at, started_at, finished_at
            FROM evolution_scan_jobs
            WHERE note_id = ? AND status IN ('queued', 'running')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (note_id,),
        ).fetchone()

    return row_to_scan_job(row) if row else None


def list_resumable_scan_jobs() -> list[dict[str, object]]:
    init_db()

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT id, note_id, status, error, attempts, max_attempts,
                   created_at, updated_at, started_at, finished_at
            FROM evolution_scan_jobs
            WHERE status IN ('queued', 'running') AND attempts < max_attempts
            ORDER BY created_at ASC
            """
        ).fetchall()

    return [row_to_scan_job(row) for row in rows]


def mark_scan_job_running(job_id: str) -> dict[str, object] | None:
    init_db()
    updated_at = now_iso()

    with connect() as connection:
        connection.execute(
            """
            UPDATE evolution_scan_jobs
            SET status = 'running',
                attempts = attempts + 1,
                error = '',
                updated_at = ?,
                started_at = ?
            WHERE id = ? AND status IN ('queued', 'running')
            """,
            (updated_at, updated_at, job_id),
        )

    return get_scan_job(job_id)


def mark_scan_job_queued(job_id: str, *, error: str = "") -> dict[str, object] | None:
    init_db()
    updated_at = now_iso()

    with connect() as connection:
        connection.execute(
            """
            UPDATE evolution_scan_jobs
            SET status = 'queued',
                error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (error, updated_at, job_id),
        )

    return get_scan_job(job_id)


def finish_scan_job(job_id: str, *, status: str, error: str = "") -> dict[str, object] | None:
    if status not in {"succeeded", "failed", "cancelled"}:
        raise ValueError(f"Unsupported scan job status: {status}")

    init_db()
    updated_at = now_iso()

    with connect() as connection:
        connection.execute(
            """
            UPDATE evolution_scan_jobs
            SET status = ?,
                error = ?,
                updated_at = ?,
                finished_at = ?
            WHERE id = ?
            """,
            (status, error, updated_at, updated_at, job_id),
        )

    return get_scan_job(job_id)
