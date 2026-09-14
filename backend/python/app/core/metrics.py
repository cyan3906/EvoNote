from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from app.core.config import settings

_CURRENT_METRICS: ContextVar[dict[str, Any] | None] = ContextVar("current_metrics", default=None)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    if not start or not end:
        return None

    return int((end - start).total_seconds() * 1000)


@contextmanager
def evolution_job(
    *,
    job_id: str,
    note_id: str,
    note_title: str,
    trigger_type: str = "scan",
    created_at: str = "",
) -> Iterator[dict[str, Any]]:
    started_at = utc_now()
    created_dt = parse_iso(created_at) or started_at
    record: dict[str, Any] = {
        "schema_version": 1,
        "event_type": "evolution_job_completed",
        "job": {
            "job_id": job_id,
            "note_id": note_id,
            "note_title": note_title,
            "trigger_type": trigger_type,
            "status": "running",
            "created_at": created_at or started_at.isoformat(),
            "started_at": started_at.isoformat(),
            "finished_at": "",
            "queue_duration_ms": duration_ms(created_dt, started_at),
            "run_duration_ms": None,
            "total_duration_ms": None,
            "error_message": "",
        },
        "counts": {
            "source_claim_count": 0,
            "candidate_note_count": 0,
            "candidate_claim_count": 0,
            "comparison_count": 0,
            "suggestion_count": 0,
            "pending_suggestion_count": 0,
            "duplicate_count": 0,
            "supplement_count": 0,
            "conflict_count": 0,
        },
        "tokens": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
        "model_calls": [],
        "api_calls": [],
        "stages": {},
        "extra": {
            "candidate_note_ids": [],
            "backend": "local",
            "retrieval_mode": "note_first_l2_vector_l3_keyword_rrf",
        },
    }
    token = _CURRENT_METRICS.set(record)
    start_perf = perf_counter()

    try:
        yield record
    except Exception as exc:
        finish_evolution_job(record, started_at, created_dt, start_perf, "failed", str(exc))
        raise
    else:
        finish_evolution_job(record, started_at, created_dt, start_perf, "succeeded", "")
    finally:
        _CURRENT_METRICS.reset(token)


def finish_evolution_job(
    record: dict[str, Any],
    started_at: datetime,
    created_at: datetime,
    start_perf: float,
    status: str,
    error_message: str,
) -> None:
    finished_at = utc_now()
    record["job"]["status"] = status
    record["job"]["finished_at"] = finished_at.isoformat()
    record["job"]["run_duration_ms"] = int((perf_counter() - start_perf) * 1000)
    record["job"]["total_duration_ms"] = duration_ms(created_at, finished_at)
    record["job"]["error_message"] = error_message
    append_record(record)


@contextmanager
def stage(name: str) -> Iterator[None]:
    start = perf_counter()

    try:
        yield
    finally:
        record = _CURRENT_METRICS.get()

        if record is not None:
            record.setdefault("stages", {})[name] = int((perf_counter() - start) * 1000)


def record_counts(**values: int) -> None:
    record = _CURRENT_METRICS.get()

    if record is None:
        return

    counts = record.setdefault("counts", {})

    for key, value in values.items():
        counts[key] = int(value)


def record_extra(**values: Any) -> None:
    record = _CURRENT_METRICS.get()

    if record is None:
        return

    record.setdefault("extra", {}).update(values)


@contextmanager
def api_call(
    *,
    operation_name: str,
    provider: str,
    model: str = "",
    call_group: str = "api_calls",
    request_size_chars: int | None = None,
) -> Iterator[dict[str, Any]]:
    started_at = utc_now()
    start = perf_counter()
    call: dict[str, Any] = {
        "operation_name": operation_name,
        "provider": provider,
        "model": model,
        "started_at": started_at.isoformat(),
        "finished_at": "",
        "duration_ms": None,
        "success": False,
        "retry_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "request_size_chars": request_size_chars or 0,
        "response_size_chars": 0,
        "error_message": "",
    }

    try:
        yield call
    except Exception as exc:
        call["error_message"] = str(exc)
        raise
    else:
        call["success"] = True
    finally:
        call["finished_at"] = utc_now_iso()
        call["duration_ms"] = int((perf_counter() - start) * 1000)
        add_call(call_group, call)


def set_call_usage(call: dict[str, Any], usage: Any) -> None:
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or input_tokens + output_tokens)
    call["input_tokens"] = input_tokens
    call["output_tokens"] = output_tokens
    call["total_tokens"] = total_tokens


def set_call_response_size(call: dict[str, Any], value: object) -> None:
    call["response_size_chars"] = len(str(value or ""))


def add_call(call_group: str, call: dict[str, Any]) -> None:
    record = _CURRENT_METRICS.get()

    if record is None:
        return

    record.setdefault(call_group, []).append(call)
    tokens = record.setdefault("tokens", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    tokens["input_tokens"] += int(call.get("input_tokens", 0) or 0)
    tokens["output_tokens"] += int(call.get("output_tokens", 0) or 0)
    tokens["total_tokens"] += int(call.get("total_tokens", 0) or 0)


def append_record(record: dict[str, Any]) -> None:
    metrics_dir = metrics_path().parent
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path().open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, indent=2))
        file.write("\n\n")


def metrics_path() -> Path:
    database_path = Path(settings.sqlite_database_path)

    if not database_path.is_absolute():
        database_path = Path(__file__).resolve().parents[2] / database_path

    return database_path.parent / "metrics" / "evolution-runs.jsonl"
