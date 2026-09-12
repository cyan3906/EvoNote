import sqlite3
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

    with connect() as connection:
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

    return get_note(note_id)


def delete_note(note_id: str) -> bool:
    init_db()

    with connect() as connection:
        cursor = connection.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    return cursor.rowcount > 0
