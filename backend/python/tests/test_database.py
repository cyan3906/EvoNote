from pathlib import Path

from app.core import database


def use_temp_database(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "evonote-test.sqlite3")


def add_analysis(note_id: str) -> None:
    database.replace_note_analysis(
        note_id=note_id,
        representation={
            "l2_summary": "summary",
            "l3_text": "retrieval text",
            "keywords": ["keyword"],
            "vector": [],
        },
        blocks=[],
        claims=[],
    )


def test_update_note_cleans_external_indexes_for_analyzed_note(monkeypatch, tmp_path: Path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(database, "_delete_external_note_indexes", lambda note_id: calls.append(note_id))
    note = database.create_note(title="old", tags="", body="old body")
    add_analysis(note["id"])

    updated = database.update_note(note["id"], title="new", tags="", body="new body")

    assert updated is not None
    assert calls == [note["id"]]


def test_update_note_skips_unchanged_content(monkeypatch, tmp_path: Path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(database, "_delete_external_note_indexes", lambda note_id: calls.append(note_id))
    note = database.create_note(title="same", tags="tag", body="same body")
    add_analysis(note["id"])

    updated = database.update_note(note["id"], title="same", tags="tag", body="same body")

    assert updated == note
    assert calls == []


def test_delete_note_cleans_external_indexes_for_analyzed_note(monkeypatch, tmp_path: Path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(database, "_delete_external_note_indexes", lambda note_id: calls.append(note_id))
    note = database.create_note(title="old", tags="", body="old body")
    add_analysis(note["id"])

    assert database.delete_note(note["id"]) is True
    assert calls == [note["id"]]


def test_scan_job_state_is_persisted(monkeypatch, tmp_path: Path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    note = database.create_note(title="job", tags="", body="body")

    job = database.create_scan_job(note["id"], max_attempts=2)
    running = database.mark_scan_job_running(str(job["id"]))

    assert running is not None
    assert running["status"] == "running"
    assert running["attempts"] == 1

    queued = database.mark_scan_job_queued(str(job["id"]), error="temporary failure")

    assert queued is not None
    assert queued["status"] == "queued"
    assert queued["error"] == "temporary failure"

    finished = database.finish_scan_job(str(job["id"]), status="succeeded")

    assert finished is not None
    assert finished["status"] == "succeeded"
    assert database.get_latest_scan_job(note["id"])["id"] == job["id"]
