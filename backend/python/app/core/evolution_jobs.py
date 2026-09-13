from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from app.core import database

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evonote-scan")
_LOCK = Lock()
_JOBS_BY_NOTE_ID: dict[str, "EvolutionScanJob"] = {}


@dataclass
class EvolutionScanJob:
    id: str
    note_id: str
    status: str
    error: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def queue_note_scan(note_id: str) -> EvolutionScanJob:
    now = _now()

    with _LOCK:
        existing = _JOBS_BY_NOTE_ID.get(note_id)

        if existing and existing.status in {"queued", "running"}:
            return existing

        job = EvolutionScanJob(
            id=str(uuid4()),
            note_id=note_id,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        _JOBS_BY_NOTE_ID[note_id] = job
        _EXECUTOR.submit(_run_scan_job, job.id, note_id)
        return job


def get_scan_job(note_id: str) -> EvolutionScanJob | None:
    with _LOCK:
        return _JOBS_BY_NOTE_ID.get(note_id)


def _run_scan_job(job_id: str, note_id: str) -> None:
    _update_job(note_id, job_id, status="running")

    try:
        from app.core.evolution import scan_note

        scan_note(note_id)
    except Exception as exc:
        _update_job(note_id, job_id, status="failed", error=str(exc))
        return

    _update_job(note_id, job_id, status="succeeded", error="")


def _update_job(note_id: str, job_id: str, *, status: str, error: str = "") -> None:
    with _LOCK:
        job = _JOBS_BY_NOTE_ID.get(note_id)

        if not job or job.id != job_id:
            return

        job.status = status
        job.error = error
        job.updated_at = _now()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
