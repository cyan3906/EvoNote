import json
from dataclasses import asdict
from typing import Any

from app.EvoRAG.entity_store.models import EntityIngestQueueResult, EntityScope, IncomingEntity
from app.EvoRAG.entity_store.normalizer import normalize_name, text_fingerprint
from app.EvoRAG.entity_store.repository_helpers import incoming_entity_key
from app.EvoRAG.entity_store.repository_mappers import (
    build_job_progress,
    incoming_status_row_to_dict,
    job_row_to_dict,
)
from app.EvoRAG.models import BlockExtractionFailure, EvoRAGPreprocessResult


class IngestJobRepositoryMixin:
    def create_ingest_job(
        self,
        *,
        input_text: str,
        preprocess: EvoRAGPreprocessResult,
        scope: EntityScope,
        incoming_entities: list[IncomingEntity],
        source_blocks_by_key: dict[str, list[dict[str, Any]]],
        source_note_id: str = "",
        task_name: str = "",
    ) -> EntityIngestQueueResult:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO evorag_ingest_jobs (
                        workspace_id, project_id, collection_id, domain,
                        source_note_id, task_name,
                        input_fingerprint, input_text, preprocess_json,
                        block_count, entity_count, queued_count, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'queued')
                    """,
                    (
                        scope.workspace_id,
                        scope.project_id,
                        scope.collection_id,
                        scope.domain,
                        str(source_note_id or ""),
                        str(task_name or "")[:255],
                        text_fingerprint(input_text),
                        input_text,
                        preprocess.model_dump_json(),
                        len(preprocess.blocks),
                        preprocess.entity_count,
                        len(incoming_entities),
                    ),
                )
                job_id = int(cursor.lastrowid)
                incoming_ids = self._insert_incoming_entities(cursor, job_id, incoming_entities, source_blocks_by_key)
                self._insert_extraction_failures(cursor, job_id, scope, source_note_id, preprocess.extraction_failures)
            connection.commit()
        return EntityIngestQueueResult(
            job_id=job_id,
            status="queued",
            queued_count=len(incoming_ids),
            incoming_entity_ids=incoming_ids,
        )

    def list_ingest_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id
                    FROM evorag_ingest_jobs
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (max(1, min(100, int(limit))),),
                )
                rows = cursor.fetchall()
        return [
            job
            for row in rows
            if (job := self.get_ingest_job_status(int(row["id"]))) is not None
        ]

    def get_ingest_job_status(self, job_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, workspace_id, project_id, collection_id, domain,
                           source_note_id, task_name,
                           block_count, entity_count, queued_count, status,
                           created_at, updated_at
                    FROM evorag_ingest_jobs
                    WHERE id = %s
                    LIMIT 1
                    """,
                    (int(job_id),),
                )
                job = cursor.fetchone()
                if not job:
                    return None

                cursor.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM evorag_incoming_entities
                    WHERE job_id = %s
                    GROUP BY status
                    """,
                    (int(job_id),),
                )
                status_rows = cursor.fetchall()
                cursor.execute(
                    """
                    SELECT
                        i.id,
                        i.name,
                        i.normalized_name,
                        i.entity_type,
                        i.status,
                        i.attempt_count,
                        i.matched_entity_id,
                        i.decision,
                        i.decision_score,
                        i.decision_reason,
                        i.last_error,
                        i.locked_by,
                        i.locked_until,
                        i.created_at,
                        i.updated_at,
                        r.id AS review_task_id,
                        r.status AS review_status
                    FROM evorag_incoming_entities i
                    LEFT JOIN evorag_entity_review_tasks r
                      ON r.incoming_entity_id = i.id AND r.status = 'pending'
                    WHERE i.job_id = %s
                    ORDER BY i.id
                    """,
                    (int(job_id),),
                )
                incoming_rows = cursor.fetchall()

        item = job_row_to_dict(job)
        item["status_counts"] = {str(row["status"]): int(row["count"]) for row in status_rows}
        item["incoming_entities"] = [incoming_status_row_to_dict(row) for row in incoming_rows]
        item["progress"] = build_job_progress(item["status_counts"], len(incoming_rows))
        return item

    def ingest_observability_summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM evorag_incoming_entities
                    GROUP BY status
                    """
                )
                status_rows = cursor.fetchall()
                cursor.execute(
                    """
                    SELECT
                        COALESCE(AVG(TIMESTAMPDIFF(MICROSECOND, created_at, updated_at)) / 1000, 0) AS avg_ms
                    FROM evorag_incoming_entities
                    WHERE status IN ('auto_merged', 'manual_merged', 'new_created', 'completed')
                    """
                )
                avg_row = cursor.fetchone() or {}
                cursor.execute(
                    """
                    SELECT
                        COALESCE(SUM(GREATEST(attempt_count - 1, 0)), 0) AS retry_count,
                        COALESCE(SUM(CASE WHEN status IN ('failed', 'dead_letter') THEN 1 ELSE 0 END), 0) AS failed_count
                    FROM evorag_incoming_entities
                    """
                )
                error_row = cursor.fetchone() or {}
        return {
            "status_counts": {str(row["status"]): int(row["count"]) for row in status_rows},
            "average_completed_ms": float(avg_row.get("avg_ms") or 0),
            "retry_count": int(error_row.get("retry_count") or 0),
            "failed_count": int(error_row.get("failed_count") or 0),
        }

    def _insert_incoming_entities(
        self,
        cursor: Any,
        job_id: int,
        incoming_entities: list[IncomingEntity],
        source_blocks_by_key: dict[str, list[dict[str, Any]]],
    ) -> list[int]:
        incoming_ids: list[int] = []
        for incoming in incoming_entities:
            key = incoming_entity_key(incoming)
            dedupe_key = text_fingerprint(f"{job_id}:{key}")
            cursor.execute(
                """
                INSERT INTO evorag_incoming_entities (
                    job_id, workspace_id, project_id, collection_id, domain,
                    dedupe_key, name, normalized_name, entity_type, aliases_json,
                    identity_description, description_for_match, attributes_json,
                    source_blocks_json, source_count, status, decision_reason, last_error
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', '', '')
                ON DUPLICATE KEY UPDATE
                    updated_at = CURRENT_TIMESTAMP,
                    id = LAST_INSERT_ID(id)
                """,
                (
                    job_id,
                    incoming.scope.workspace_id,
                    incoming.scope.project_id,
                    incoming.scope.collection_id,
                    incoming.scope.domain,
                    dedupe_key,
                    incoming.name,
                    incoming.normalized_name or normalize_name(incoming.name),
                    incoming.entity_type or "concept",
                    json.dumps(incoming.aliases, ensure_ascii=False),
                    incoming.identity_description,
                    incoming.description_for_match,
                    json.dumps([asdict(attribute) for attribute in incoming.attributes], ensure_ascii=False),
                    json.dumps(source_blocks_by_key.get(key, []), ensure_ascii=False),
                    incoming.source_count,
                ),
            )
            incoming_ids.append(int(cursor.lastrowid))
        return incoming_ids

    def _insert_extraction_failures(
        self,
        cursor: Any,
        job_id: int,
        scope: EntityScope,
        source_note_id: str,
        failures: list[BlockExtractionFailure],
    ) -> None:
        for failure in failures:
            cursor.execute(
                """
                INSERT INTO evorag_entity_extraction_failures (
                    job_id, workspace_id, project_id, collection_id, domain,
                    source_note_id, block_index, heading, anchor_entity,
                    l1_text, error, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
                """,
                (
                    int(job_id),
                    scope.workspace_id,
                    scope.project_id,
                    scope.collection_id,
                    scope.domain,
                    str(source_note_id or ""),
                    int(failure.block_index),
                    failure.heading[:255],
                    failure.anchor_entity[:255],
                    failure.l1_text,
                    failure.error,
                ),
            )
