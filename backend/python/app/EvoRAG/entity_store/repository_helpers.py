from typing import Any

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.entity_store.models import EntityScope, IncomingEntity
from app.EvoRAG.entity_store.normalizer import normalize_name


def update_ingest_job_status(cursor: Any, job_id: int) -> None:
    cursor.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM evorag_incoming_entities
        WHERE job_id = %s
        GROUP BY status
        """,
        (job_id,),
    )
    counts = {str(row["status"]): int(row["count"]) for row in cursor.fetchall()}
    if counts.get("pending", 0) or counts.get("processing", 0):
        status = "processing"
    elif counts.get("dead_letter", 0) or counts.get("failed", 0):
        status = "failed"
    elif counts.get("needs_review", 0):
        status = "needs_review"
    else:
        status = "completed"
    cursor.execute(
        """
        UPDATE evorag_ingest_jobs
        SET status = %s,
            queued_count = %s
        WHERE id = %s
        """,
        (status, sum(counts.values()), job_id),
    )


def scope_from_config(config: EvoRAGSettings) -> EntityScope:
    return EntityScope(
        workspace_id=config.default_workspace_id,
        project_id=config.default_project_id,
        collection_id=config.default_collection_id,
        domain=config.default_domain,
    )


def build_scope_where(scope: EntityScope) -> tuple[str, tuple[str, str, str, str]]:
    return (
        "AND workspace_id = %s AND project_id = %s AND collection_id = %s AND domain = %s",
        (scope.workspace_id, scope.project_id, scope.collection_id, scope.domain),
    )


def unique_ints(values: list[int]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def unique_strings(values: list[str], *, sort: bool) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return sorted(result) if sort else result


def incoming_entity_key(incoming: IncomingEntity) -> str:
    return f"{normalize_name(incoming.name)}::{normalize_name(incoming.entity_type)}"
