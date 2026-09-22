from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any

from app.core import database
from app.core.config import settings

_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, settings.evolution_job_max_workers), thread_name_prefix="evonote-scan")
_LOCK = Lock()
_SUBMITTED_JOB_IDS: set[str] = set()


@dataclass
class EvolutionScanJob:
    id: str
    note_id: str
    status: str
    error: str = ""
    created_at: str = ""
    updated_at: str = ""
    attempts: int = 0
    max_attempts: int = 3
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def queue_note_scan(note_id: str) -> EvolutionScanJob:
    with _LOCK:
        job = database.get_active_scan_job(note_id)

        if job is None:
            job = database.create_scan_job(note_id, max_attempts=settings.evolution_job_max_attempts)

        _submit_job_locked(str(job["id"]))
        return _job_from_dict(job)


def get_scan_job(note_id: str) -> EvolutionScanJob | None:
    job = database.get_latest_scan_job(note_id)
    return _job_from_dict(job) if job else None


def recover_pending_scan_jobs() -> list[EvolutionScanJob]:
    recovered: list[EvolutionScanJob] = []

    with _LOCK:
        for job in database.list_resumable_scan_jobs():
            if job["status"] == "running":
                job = database.mark_scan_job_queued(str(job["id"]), error="任务在服务重启后恢复排队") or job

            _submit_job_locked(str(job["id"]))
            recovered.append(_job_from_dict(job))

    return recovered


def shutdown_scan_jobs() -> None:
    _EXECUTOR.shutdown(wait=False, cancel_futures=False)


def _submit_job_locked(job_id: str) -> None:
    if job_id in _SUBMITTED_JOB_IDS:
        return

    _SUBMITTED_JOB_IDS.add(job_id)
    _EXECUTOR.submit(_run_scan_job, job_id)


def _run_scan_job(job_id: str) -> None:
    job = database.mark_scan_job_running(job_id)

    if not job:
        _forget_submitted(job_id)
        return

    try:
        from app.core.evolution import scan_note

        scan_note(
            str(job["note_id"]),
            job_id=job_id,
            created_at=str(job.get("created_at", "")),
            trigger_type="background_scan",
        )
    except Exception as exc:
        _handle_job_failure(job, str(exc))
        return

    database.finish_scan_job(job_id, status="succeeded", error="")
    _forget_submitted(job_id)


def _handle_job_failure(job: dict[str, object], error: str) -> None:
    job_id = str(job["id"])
    attempts = int(job.get("attempts", 0) or 0)
    max_attempts = int(job.get("max_attempts", settings.evolution_job_max_attempts) or settings.evolution_job_max_attempts)

    if attempts < max_attempts:
        database.mark_scan_job_queued(job_id, error=error)
        _forget_submitted(job_id)

        with _LOCK:
            _submit_job_locked(job_id)
        return

    database.finish_scan_job(job_id, status="failed", error=error)
    _forget_submitted(job_id)


def _forget_submitted(job_id: str) -> None:
    with _LOCK:
        _SUBMITTED_JOB_IDS.discard(job_id)


def _job_from_dict(job: dict[str, object]) -> EvolutionScanJob:
    return EvolutionScanJob(
        id=str(job["id"]),
        note_id=str(job["note_id"]),
        status=str(job["status"]),
        error=str(job.get("error", "")),
        created_at=str(job.get("created_at", "")),
        updated_at=str(job.get("updated_at", "")),
        attempts=int(job.get("attempts", 0) or 0),
        max_attempts=int(job.get("max_attempts", settings.evolution_job_max_attempts) or settings.evolution_job_max_attempts),
        started_at=str(job.get("started_at", "")),
        finished_at=str(job.get("finished_at", "")),
    )
