import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.alias_cache import EntityAliasCache, scope_hash
from app.EvoRAG.entity_store.models import CandidateEntity, EntityAttributeInput, EntityIngestQueueResult, EntityRelationMemoryRecord, EntityScope, EntityUpsertResult, IncomingEntity, IncomingEntityTask, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name, text_fingerprint
from app.EvoRAG.models import EvoRAGPreprocessResult, StoredAttribute, StoredAttributeEvidence
from app.core.evorag_database import connect_evorag_mysql


class MySQLEntityRepository:
    def __init__(self, config: EvoRAGSettings = settings, alias_cache: EntityAliasCache | None = None) -> None:
        self.config = config
        self.alias_cache = alias_cache or EntityAliasCache()

    def init_schema(self) -> None:
        sql_path = Path(__file__).resolve().parents[1] / "sql" / "entity_schema.sql"
        statements = [statement.strip() for statement in sql_path.read_text(encoding="utf-8").split(";") if statement.strip()]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
                ensure_entity_table_columns(cursor)
            connection.commit()
        if not self.alias_table_has_rows():
            self.backfill_entity_aliases()

    def list_entities(self, scope: EntityScope | None = None) -> list[StoredEntity]:
        scope = scope or scope_from_config(self.config)
        where_sql, values = build_scope_where(scope)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT id, workspace_id, project_id, collection_id, domain,
                           canonical_name, normalized_name, entity_type, aliases_json,
                           identity_description, summary, description_for_match, embedding_json
                    FROM evorag_entities
                    WHERE status = 'active' {where_sql}
                    """,
                    values,
                )
                rows = cursor.fetchall()
        return [row_to_entity(row) for row in rows]

    def list_all_entities(self) -> list[StoredEntity]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, workspace_id, project_id, collection_id, domain,
                           canonical_name, normalized_name, entity_type, aliases_json,
                           identity_description, summary, description_for_match, embedding_json
                    FROM evorag_entities
                    WHERE status = 'active'
                    """
                )
                rows = cursor.fetchall()
        return [row_to_entity(row) for row in rows]

    def find_by_normalized_name(
        self,
        normalized_name: str,
        entity_type: str = "",
        scope: EntityScope | None = None,
    ) -> StoredEntity | None:
        entity_scope = scope or scope_from_config(self.config)
        where_sql, scope_values = build_scope_where(entity_scope)
        sql = f"""
            SELECT id, workspace_id, project_id, collection_id, domain,
                   canonical_name, normalized_name, entity_type, aliases_json,
                   identity_description, summary, description_for_match, embedding_json
            FROM evorag_entities
            WHERE normalized_name = %s AND status = 'active' {where_sql}
        """
        values: list[Any] = [normalized_name, *scope_values]
        if entity_type:
            sql += " AND entity_type = %s"
            values.append(entity_type)
        sql += " LIMIT 1"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, tuple(values))
                row = cursor.fetchone()
        return row_to_entity(row) if row else None

    def get_entity(self, entity_id: int) -> StoredEntity | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, workspace_id, project_id, collection_id, domain,
                           canonical_name, normalized_name, entity_type, aliases_json,
                           identity_description, summary, description_for_match, embedding_json
                    FROM evorag_entities
                    WHERE id = %s AND status = 'active'
                    LIMIT 1
                    """,
                    (entity_id,),
                )
                row = cursor.fetchone()
        return row_to_entity(row) if row else None

    def find_by_alias(self, alias: str, scope: EntityScope | None = None) -> StoredEntity | None:
        normalized_alias = normalize_name(alias)
        if not normalized_alias:
            return None

        entity_scope = scope or scope_from_config(self.config)
        cached_entity_id = self.alias_cache.get_entity_id(entity_scope, normalized_alias)
        if cached_entity_id:
            cached = self.get_entity(cached_entity_id)
            if cached is not None:
                return cached

        entity_id = self.find_alias_entity_id(normalized_alias, entity_scope)
        if not entity_id:
            return None
        self.alias_cache.set_entity_id(entity_scope, normalized_alias, entity_id)
        return self.get_entity(entity_id)

    def find_alias_for_incoming(self, incoming: IncomingEntity) -> StoredEntity | None:
        aliases = [incoming.name, incoming.normalized_name, *incoming.aliases]
        for alias in aliases:
            entity = self.find_by_alias(alias, incoming.scope)
            if entity is None:
                continue
            if incoming.entity_type and entity.entity_type and incoming.entity_type != entity.entity_type:
                continue
            return entity
        return None

    def find_alias_entity_id(self, normalized_alias: str, scope: EntityScope | None = None) -> int | None:
        entity_scope = scope or scope_from_config(self.config)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT entity_id
                    FROM evorag_entity_aliases
                    WHERE normalized_alias = %s
                      AND scope_hash = %s
                      AND status = 'active'
                    LIMIT 1
                    """,
                    (normalized_alias, scope_hash(entity_scope)),
                )
                row = cursor.fetchone()
        return int(row["entity_id"]) if row else None

    def upsert_entity_alias(
        self,
        *,
        entity_id: int,
        alias_text: str,
        scope: EntityScope,
        alias_type: str = "alias",
        confidence: float = 1.0,
        source: str = "system",
    ) -> None:
        normalized_alias = normalize_name(alias_text)
        clean_alias = " ".join(str(alias_text or "").split())
        if not normalized_alias or not clean_alias or not entity_id:
            return

        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO evorag_entity_aliases (
                        workspace_id, project_id, collection_id, domain, scope_hash,
                        entity_id, alias_text, normalized_alias, alias_type, confidence, source, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
                    ON DUPLICATE KEY UPDATE
                        entity_id = VALUES(entity_id),
                        alias_text = VALUES(alias_text),
                        alias_type = VALUES(alias_type),
                        confidence = GREATEST(confidence, VALUES(confidence)),
                        source = VALUES(source),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        scope.workspace_id,
                        scope.project_id,
                        scope.collection_id,
                        scope.domain,
                        scope_hash(scope),
                        entity_id,
                        clean_alias,
                        normalized_alias,
                        alias_type,
                        max(0.0, min(1.0, float(confidence))),
                        source,
                    ),
                )
            connection.commit()
        self.alias_cache.set_entity_id(scope, normalized_alias, entity_id)

    def upsert_aliases_for_entity(self, entity: StoredEntity, *, source: str = "system", confidence: float = 1.0) -> None:
        self.upsert_entity_alias(
            entity_id=entity.id,
            alias_text=entity.canonical_name,
            scope=entity.scope,
            alias_type="canonical",
            confidence=confidence,
            source=source,
        )
        for alias in entity.aliases:
            self.upsert_entity_alias(
                entity_id=entity.id,
                alias_text=alias,
                scope=entity.scope,
                alias_type="alias",
                confidence=confidence,
                source=source,
            )

    def backfill_entity_aliases(self) -> None:
        for entity in self.list_all_entities():
            self.upsert_aliases_for_entity(entity, source="schema_backfill", confidence=1.0)

    def alias_table_has_rows(self) -> bool:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1 FROM evorag_entity_aliases WHERE status = 'active' LIMIT 1")
                return cursor.fetchone() is not None

    def hydrate_entities_for_candidates(
        self,
        entity_ids: list[int],
    ) -> tuple[dict[int, StoredEntity], dict[int, list[StoredAttribute]]]:
        unique_ids = unique_ints(entity_ids)
        if not unique_ids:
            return {}, {}

        placeholders = ", ".join(["%s"] * len(unique_ids))
        entity_sql = f"""
            SELECT id, workspace_id, project_id, collection_id, domain,
                   canonical_name, normalized_name, entity_type, aliases_json,
                   identity_description, summary, description_for_match, embedding_json
            FROM evorag_entities
            WHERE id IN ({placeholders}) AND status = 'active'
        """
        attribute_sql = f"""
            SELECT
                a.id AS attribute_id,
                a.entity_id,
                a.attr_type,
                a.value_text,
                a.confidence,
                e.evidence_text,
                e.note_id,
                e.block_id,
                e.block_index
            FROM evorag_entity_attributes a
            LEFT JOIN evorag_entity_attribute_evidence e ON e.attribute_id = a.id
            WHERE a.entity_id IN ({placeholders}) AND a.status = 'active'
            ORDER BY a.entity_id, a.attr_type, a.id, e.id
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(entity_sql, tuple(unique_ids))
                entity_rows = cursor.fetchall()
                cursor.execute(attribute_sql, tuple(unique_ids))
                attribute_rows = cursor.fetchall()

        entities = {entity.id: entity for entity in (row_to_entity(row) for row in entity_rows)}
        return entities, group_attribute_rows(attribute_rows)

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

    def list_attributes_for_entities(self, entity_ids: list[int]) -> dict[int, list[StoredAttribute]]:
        if not entity_ids:
            return {}

        placeholders = ", ".join(["%s"] * len(entity_ids))
        sql = f"""
            SELECT
                a.id AS attribute_id,
                a.entity_id,
                a.attr_type,
                a.value_text,
                a.confidence,
                e.evidence_text,
                e.note_id,
                e.block_id,
                e.block_index
            FROM evorag_entity_attributes a
            LEFT JOIN evorag_entity_attribute_evidence e ON e.attribute_id = a.id
            WHERE a.entity_id IN ({placeholders}) AND a.status = 'active'
            ORDER BY a.entity_id, a.attr_type, a.id, e.id
        """
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, tuple(entity_ids))
                rows = cursor.fetchall()
        return group_attribute_rows(rows)

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

    def list_resumable_incoming_entity_ids(self, *, limit: int = 200) -> list[int]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id
                    FROM evorag_incoming_entities
                    WHERE status = 'pending'
                       OR (status = 'processing' AND (locked_until IS NULL OR locked_until < CURRENT_TIMESTAMP))
                    ORDER BY id
                    LIMIT %s
                    """,
                    (max(1, int(limit)),),
                )
                rows = cursor.fetchall()
        return [int(row["id"]) for row in rows]

    def get_incoming_entity_task(self, incoming_entity_id: int) -> IncomingEntityTask | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, job_id, workspace_id, project_id, collection_id, domain,
                           name, normalized_name, entity_type, aliases_json,
                           identity_description, description_for_match, attributes_json,
                           source_count, status, attempt_count
                    FROM evorag_incoming_entities
                    WHERE id = %s
                    LIMIT 1
                    """,
                    (incoming_entity_id,),
                )
                row = cursor.fetchone()
        return row_to_incoming_task(row) if row else None

    def claim_incoming_entity(self, incoming_entity_id: int, *, worker_id: str, lock_seconds: int) -> IncomingEntityTask | None:
        lock_seconds = max(1, int(lock_seconds))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    UPDATE evorag_incoming_entities
                    SET status = 'processing',
                        attempt_count = attempt_count + 1,
                        locked_by = %s,
                        locked_until = DATE_ADD(CURRENT_TIMESTAMP, INTERVAL {lock_seconds} SECOND),
                        last_error = ''
                    WHERE id = %s
                      AND (
                        status = 'pending'
                        OR (status = 'processing' AND (locked_until IS NULL OR locked_until < CURRENT_TIMESTAMP))
                      )
                    """,
                    (worker_id, incoming_entity_id),
                )
                changed = cursor.rowcount
            connection.commit()
        if not changed:
            return None
        return self.get_incoming_entity_task(incoming_entity_id)

    def finish_incoming_entity(
        self,
        incoming_entity_id: int,
        *,
        status: str,
        decision: str = "",
        matched_entity_id: int | None = None,
        score: float = 0.0,
        reason: str = "",
        error: str = "",
    ) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE evorag_incoming_entities
                    SET status = %s,
                        matched_entity_id = %s,
                        decision = %s,
                        decision_score = %s,
                        decision_reason = %s,
                        last_error = %s,
                        locked_by = '',
                        locked_until = NULL
                    WHERE id = %s
                    """,
                    (
                        status,
                        matched_entity_id,
                        decision,
                        score,
                        reason,
                        error,
                        incoming_entity_id,
                    ),
                )
                cursor.execute("SELECT job_id FROM evorag_incoming_entities WHERE id = %s", (incoming_entity_id,))
                row = cursor.fetchone()
                job_id = int(row["job_id"]) if row else 0
                if job_id:
                    update_ingest_job_status(cursor, job_id)
            connection.commit()

    def mark_incoming_failure(self, incoming_entity_id: int, *, error: str, max_attempts: int) -> str:
        task = self.get_incoming_entity_task(incoming_entity_id)
        if task is None:
            return "missing"
        status = "pending" if task.attempt_count < max(1, int(max_attempts)) else "dead_letter"
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE evorag_incoming_entities
                    SET status = %s,
                        last_error = %s,
                        locked_by = '',
                        locked_until = NULL
                    WHERE id = %s
                    """,
                    (status, error, incoming_entity_id),
                )
                update_ingest_job_status(cursor, task.job_id)
            connection.commit()
        return status

    def rejected_candidate_ids(self, incoming: IncomingEntity) -> set[int]:
        normalized_name = incoming.normalized_name or normalize_name(incoming.name)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT candidate_entity_id
                    FROM evorag_entity_resolution_rejections
                    WHERE scope_hash = %s
                      AND incoming_normalized_name = %s
                      AND incoming_type = %s
                      AND status = 'active'
                    """,
                    (scope_hash(incoming.scope), normalized_name, incoming.entity_type or ""),
                )
                rows = cursor.fetchall()
        return {int(row["candidate_entity_id"]) for row in rows}

    def list_entity_resolution_relation_memory(
        self,
        incoming: IncomingEntity,
        *,
        limit: int = 30,
    ) -> list[EntityRelationMemoryRecord]:
        normalized_name = incoming.normalized_name or normalize_name(incoming.name)
        entity_type = incoming.entity_type or ""
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, left_entity_id, left_name, left_normalized_name, left_type,
                           right_entity_id, right_name, right_normalized_name, right_type,
                           decision, relation_type, confidence, hit_count, source, reason
                    FROM evorag_entity_resolution_relation_memory
                    WHERE scope_hash = %s
                      AND status = 'active'
                      AND (
                            (left_normalized_name = %s AND (left_type = %s OR %s = ''))
                         OR (right_normalized_name = %s AND (right_type = %s OR %s = ''))
                      )
                    ORDER BY hit_count DESC, confidence DESC, updated_at DESC, id DESC
                    LIMIT %s
                    """,
                    (
                        scope_hash(incoming.scope),
                        normalized_name,
                        entity_type,
                        entity_type,
                        normalized_name,
                        entity_type,
                        entity_type,
                        max(1, int(limit)),
                    ),
                )
                rows = cursor.fetchall()

        records: list[EntityRelationMemoryRecord] = []
        for row in rows:
            record = self._relation_memory_record_from_row(row, incoming)
            if record is not None:
                records.append(record)
        return records

    def _relation_memory_record_from_row(
        self,
        row: dict[str, Any],
        incoming: IncomingEntity,
    ) -> EntityRelationMemoryRecord | None:
        normalized_name = incoming.normalized_name or normalize_name(incoming.name)
        incoming_type = incoming.entity_type or ""
        left_matches = str(row["left_normalized_name"]) == normalized_name and (not incoming_type or str(row["left_type"]) == incoming_type)
        candidate_side = "right" if left_matches else "left"
        candidate_entity_id = row.get(f"{candidate_side}_entity_id")
        if candidate_entity_id is None:
            return None

        entity = self.get_entity(int(candidate_entity_id))
        if entity is None:
            entity = StoredEntity(
                id=int(candidate_entity_id),
                canonical_name=str(row.get(f"{candidate_side}_name") or ""),
                normalized_name=str(row.get(f"{candidate_side}_normalized_name") or ""),
                entity_type=str(row.get(f"{candidate_side}_type") or ""),
                scope=incoming.scope,
            )
        return EntityRelationMemoryRecord(
            id=int(row["id"]),
            decision=str(row.get("decision") or ""),
            relation_type=str(row.get("relation_type") or ""),
            candidate=entity,
            confidence=float(row.get("confidence") or 0.0),
            hit_count=int(row.get("hit_count") or 0),
            source=str(row.get("source") or ""),
            reason=str(row.get("reason") or ""),
        )

    def upsert_entity_resolution_relation_memory(
        self,
        *,
        incoming: IncomingEntity,
        candidate: StoredEntity,
        decision: str,
        relation_type: str,
        confidence: float,
        source: str,
        reason: str,
    ) -> None:
        normalized_name = incoming.normalized_name or normalize_name(incoming.name)
        candidate_normalized_name = candidate.normalized_name or normalize_name(candidate.canonical_name)
        clean_decision = str(decision or "").strip().lower()
        if clean_decision not in {"allow", "reject"}:
            raise ValueError(f"unsupported relation memory decision: {decision}")
        candidate_entity_id = int(candidate.id) if self.get_entity(int(candidate.id)) is not None else None
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO evorag_entity_resolution_relation_memory (
                        workspace_id, project_id, collection_id, domain, scope_hash,
                        left_entity_id, left_name, left_normalized_name, left_type,
                        right_entity_id, right_name, right_normalized_name, right_type,
                        decision, relation_type, confidence, hit_count, source, reason, status
                    )
                    VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s, %s, 'active')
                    ON DUPLICATE KEY UPDATE
                        decision = VALUES(decision),
                        relation_type = VALUES(relation_type),
                        confidence = VALUES(confidence),
                        hit_count = hit_count + 1,
                        source = VALUES(source),
                        reason = VALUES(reason),
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        incoming.scope.workspace_id,
                        incoming.scope.project_id,
                        incoming.scope.collection_id,
                        incoming.scope.domain,
                        scope_hash(incoming.scope),
                        incoming.name,
                        normalized_name,
                        incoming.entity_type or "",
                        candidate_entity_id,
                        candidate.canonical_name,
                        candidate_normalized_name,
                        candidate.entity_type or "",
                        clean_decision,
                        relation_type,
                        max(0.0, min(1.0, float(confidence))),
                        source,
                        reason,
                    ),
                )
            connection.commit()

    def create_review_task(
        self,
        *,
        incoming_entity_id: int,
        task: IncomingEntityTask,
        candidates: list[CandidateEntity],
        reason: str,
    ) -> int:
        incoming_json = json.dumps(incoming_snapshot(task.incoming), ensure_ascii=False)
        candidates_json = json.dumps([candidate_snapshot(candidate) for candidate in candidates], ensure_ascii=False)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id
                    FROM evorag_entity_review_tasks
                    WHERE incoming_entity_id = %s AND status = 'pending'
                    LIMIT 1
                    """,
                    (incoming_entity_id,),
                )
                existing = cursor.fetchone()
                if existing:
                    review_id = int(existing["id"])
                    cursor.execute(
                        """
                        UPDATE evorag_entity_review_tasks
                        SET incoming_snapshot_json = %s,
                            candidates_json = %s,
                            reason = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (incoming_json, candidates_json, reason, review_id),
                    )
                    connection.commit()
                    return review_id

                cursor.execute(
                    """
                    INSERT INTO evorag_entity_review_tasks (
                        incoming_entity_id, job_id,
                        workspace_id, project_id, collection_id, domain,
                        incoming_snapshot_json, candidates_json, status, reason
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending', %s)
                    """,
                    (
                        incoming_entity_id,
                        task.job_id,
                        task.incoming.scope.workspace_id,
                        task.incoming.scope.project_id,
                        task.incoming.scope.collection_id,
                        task.incoming.scope.domain,
                        incoming_json,
                        candidates_json,
                        reason,
                    ),
                )
                review_id = int(cursor.lastrowid)
            connection.commit()
        return review_id

    def list_review_tasks(self, *, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, incoming_entity_id, job_id,
                           workspace_id, project_id, collection_id, domain,
                           incoming_snapshot_json, candidates_json, status,
                           decision, decided_entity_id, decided_by, reason,
                           created_at, updated_at
                    FROM evorag_entity_review_tasks
                    WHERE status = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (status, max(1, int(limit))),
                )
                rows = cursor.fetchall()
        return [review_row_to_dict(row) for row in rows]

    def get_review_task(self, review_task_id: int) -> dict[str, Any] | None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, incoming_entity_id, job_id,
                           workspace_id, project_id, collection_id, domain,
                           incoming_snapshot_json, candidates_json, status,
                           decision, decided_entity_id, decided_by, reason,
                           created_at, updated_at
                    FROM evorag_entity_review_tasks
                    WHERE id = %s
                    LIMIT 1
                    """,
                    (review_task_id,),
                )
                row = cursor.fetchone()
        return review_row_to_dict(row) if row else None

    def complete_review_task(
        self,
        review_task_id: int,
        *,
        status: str,
        decision: str,
        decided_entity_id: int | None = None,
        decided_by: str = "manual",
        reason: str = "",
    ) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE evorag_entity_review_tasks
                    SET status = %s,
                        decision = %s,
                        decided_entity_id = %s,
                        decided_by = %s,
                        reason = %s,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (status, decision, decided_entity_id, decided_by, reason, review_task_id),
                )
            connection.commit()

    def record_rejections(
        self,
        *,
        incoming: IncomingEntity,
        candidate_entity_ids: list[int],
        reason: str,
        decided_by: str = "manual",
    ) -> None:
        if not candidate_entity_ids:
            return
        normalized_name = incoming.normalized_name or normalize_name(incoming.name)
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for candidate_id in candidate_entity_ids:
                    cursor.execute(
                        """
                        INSERT INTO evorag_entity_resolution_rejections (
                            workspace_id, project_id, collection_id, domain, scope_hash,
                            incoming_normalized_name, incoming_type, candidate_entity_id,
                            reason, decided_by, status
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active')
                        ON DUPLICATE KEY UPDATE
                            reason = VALUES(reason),
                            decided_by = VALUES(decided_by),
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            incoming.scope.workspace_id,
                            incoming.scope.project_id,
                            incoming.scope.collection_id,
                            incoming.scope.domain,
                            scope_hash(incoming.scope),
                            normalized_name,
                            incoming.entity_type or "",
                            int(candidate_id),
                            reason,
                            decided_by,
                        ),
                    )
            connection.commit()

    def reset_incoming_for_retry(self, incoming_entity_id: int) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE evorag_incoming_entities
                    SET status = 'pending',
                        locked_by = '',
                        locked_until = NULL,
                        last_error = ''
                    WHERE id = %s
                    """,
                    (incoming_entity_id,),
                )
                cursor.execute("SELECT job_id FROM evorag_incoming_entities WHERE id = %s", (incoming_entity_id,))
                row = cursor.fetchone()
                if row:
                    update_ingest_job_status(cursor, int(row["job_id"]))
            connection.commit()

    def upsert_entity(self, incoming: IncomingEntity, *, matched_entity_id: int | None = None) -> EntityUpsertResult:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                if matched_entity_id:
                    entity_id = matched_entity_id
                    cursor.execute(
                        """
                        UPDATE evorag_entities
                        SET aliases_json = %s,
                            workspace_id = %s,
                            project_id = %s,
                            collection_id = %s,
                            domain = %s,
                            identity_description = %s,
                            description_for_match = %s,
                            embedding_json = %s
                        WHERE id = %s
                        """,
                        (
                            json.dumps(incoming.aliases, ensure_ascii=False),
                            incoming.scope.workspace_id,
                            incoming.scope.project_id,
                            incoming.scope.collection_id,
                            incoming.scope.domain,
                            incoming.identity_description,
                            incoming.description_for_match,
                            json.dumps(incoming.embedding, ensure_ascii=False),
                            entity_id,
                        ),
                    )
                    created = False
                else:
                    cursor.execute(
                        """
                        INSERT INTO evorag_entities (
                            workspace_id, project_id, collection_id, domain,
                            canonical_name, normalized_name, entity_type, aliases_json,
                            identity_description, summary, description_for_match, embedding_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            incoming.scope.workspace_id,
                            incoming.scope.project_id,
                            incoming.scope.collection_id,
                            incoming.scope.domain,
                            incoming.name,
                            incoming.normalized_name or normalize_name(incoming.name),
                            incoming.entity_type or "concept",
                            json.dumps(incoming.aliases, ensure_ascii=False),
                            incoming.identity_description,
                            build_summary(incoming),
                            incoming.description_for_match,
                            json.dumps(incoming.embedding, ensure_ascii=False),
                        ),
                    )
                    entity_id = int(cursor.lastrowid)
                    created = True

                attribute_count, evidence_count = self._merge_attributes(cursor, entity_id, incoming.attributes)
            connection.commit()

        return EntityUpsertResult(
            entity_id=entity_id,
            canonical_name=incoming.name,
            created=created,
            attribute_count=attribute_count,
            evidence_count=evidence_count,
        )

    def record_resolution_audit(self, incoming: IncomingEntity, decision: str, matched_entity_id: int | None, score: float, reason: str) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO evorag_entity_resolution_audit (
                        workspace_id, project_id, collection_id, domain,
                        incoming_name, incoming_type, decision, matched_entity_id, score, reason
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        incoming.scope.workspace_id,
                        incoming.scope.project_id,
                        incoming.scope.collection_id,
                        incoming.scope.domain,
                        incoming.name,
                        incoming.entity_type,
                        decision,
                        matched_entity_id,
                        score,
                        reason,
                    ),
                )
            connection.commit()

    def _merge_attributes(self, cursor: Any, entity_id: int, attributes: list[EntityAttributeInput]) -> tuple[int, int]:
        attribute_count = 0
        evidence_count = 0
        for attribute in attributes:
            value_fingerprint = text_fingerprint(attribute.value_text)
            cursor.execute(
                """
                INSERT INTO evorag_entity_attributes (
                    entity_id, attr_type, value_text, value_fingerprint, confidence
                )
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    confidence = GREATEST(confidence, VALUES(confidence)),
                    updated_at = CURRENT_TIMESTAMP
                """,
                (entity_id, attribute.attr_type, attribute.value_text, value_fingerprint, attribute.confidence),
            )
            attribute_count += 1
            cursor.execute(
                """
                SELECT id FROM evorag_entity_attributes
                WHERE entity_id = %s AND attr_type = %s AND value_fingerprint = %s
                LIMIT 1
                """,
                (entity_id, attribute.attr_type, value_fingerprint),
            )
            row = cursor.fetchone()
            if not row:
                continue
            attribute_id = int(row["id"])
            if attribute.evidence:
                cursor.execute(
                    """
                    INSERT INTO evorag_entity_attribute_evidence (
                        attribute_id, note_id, block_id, block_index, evidence_text, evidence_fingerprint
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        evidence_text = VALUES(evidence_text)
                    """,
                    (
                        attribute_id,
                        attribute.note_id,
                        attribute.block_id,
                        attribute.block_index,
                        attribute.evidence,
                        text_fingerprint(attribute.evidence),
                    ),
                )
                evidence_count += 1
        return attribute_count, evidence_count

    def connect(self):
        return connect_evorag_mysql(self.config)


def row_to_entity(row: dict[str, Any]) -> StoredEntity:
    return StoredEntity(
        id=int(row["id"]),
        canonical_name=str(row["canonical_name"]),
        normalized_name=str(row["normalized_name"]),
        entity_type=str(row["entity_type"]),
        scope=EntityScope(
            workspace_id=str(row.get("workspace_id") or "local"),
            project_id=str(row.get("project_id") or "evorag"),
            collection_id=str(row.get("collection_id") or "default"),
            domain=str(row.get("domain") or "general"),
        ),
        aliases=json.loads(row.get("aliases_json") or "[]"),
        identity_description=str(row.get("identity_description") or ""),
        summary=str(row.get("summary") or ""),
        description_for_match=str(row.get("description_for_match") or ""),
        embedding=json.loads(row.get("embedding_json") or "[]"),
    )


def row_to_incoming_task(row: dict[str, Any]) -> IncomingEntityTask:
    attributes = [
        EntityAttributeInput(
            attr_type=str(item.get("attr_type") or ""),
            value_text=str(item.get("value_text") or ""),
            evidence=str(item.get("evidence") or ""),
            confidence=float(item.get("confidence") or 0.7),
            note_id=str(item.get("note_id") or ""),
            block_id=str(item.get("block_id") or ""),
            block_index=int(item.get("block_index") if item.get("block_index") is not None else -1),
        )
        for item in json.loads(row.get("attributes_json") or "[]")
        if isinstance(item, dict)
    ]
    incoming = IncomingEntity(
        name=str(row["name"]),
        normalized_name=str(row["normalized_name"]),
        entity_type=str(row["entity_type"] or "concept"),
        scope=EntityScope(
            workspace_id=str(row.get("workspace_id") or "local"),
            project_id=str(row.get("project_id") or "evorag"),
            collection_id=str(row.get("collection_id") or "default"),
            domain=str(row.get("domain") or "general"),
        ),
        aliases=json.loads(row.get("aliases_json") or "[]"),
        identity_description=str(row.get("identity_description") or ""),
        attributes=attributes,
        description_for_match=str(row.get("description_for_match") or ""),
        source_count=int(row.get("source_count") or 1),
    )
    return IncomingEntityTask(
        id=int(row["id"]),
        job_id=int(row["job_id"]),
        incoming=incoming,
        status=str(row.get("status") or ""),
        attempt_count=int(row.get("attempt_count") or 0),
    )


def incoming_snapshot(incoming: IncomingEntity) -> dict[str, Any]:
    return {
        "name": incoming.name,
        "normalized_name": incoming.normalized_name,
        "entity_type": incoming.entity_type,
        "aliases": incoming.aliases,
        "identity_description": incoming.identity_description,
        "description_for_match": incoming.description_for_match,
        "source_count": incoming.source_count,
        "attributes": [asdict(attribute) for attribute in incoming.attributes],
        "scope": incoming.scope.as_dict(),
    }


def candidate_snapshot(candidate: CandidateEntity) -> dict[str, Any]:
    entity = candidate.entity
    return {
        "id": entity.id,
        "canonical_name": entity.canonical_name,
        "normalized_name": entity.normalized_name,
        "entity_type": entity.entity_type,
        "aliases": entity.aliases,
        "identity_description": entity.identity_description,
        "summary": entity.summary,
        "description_for_match": entity.description_for_match,
        "score": candidate.score,
        "rank": candidate.rank,
        "source": candidate.source,
        "vector_score": candidate.vector_score,
        "es_score": candidate.es_score,
    }


def review_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["incoming_snapshot"] = json.loads(item.pop("incoming_snapshot_json") or "{}")
    item["candidates"] = json.loads(item.pop("candidates_json") or "[]")
    for key in ("created_at", "updated_at"):
        if item.get(key) is not None:
            item[key] = str(item[key])
    return item


def job_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    for key in ("created_at", "updated_at"):
        if item.get(key) is not None:
            item[key] = str(item[key])
    return item


def incoming_status_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    for key in ("created_at", "updated_at", "locked_until"):
        if item.get(key) is not None:
            item[key] = str(item[key])
    if item.get("review_task_id") is not None:
        item["review_task_id"] = int(item["review_task_id"])
    return item


def build_job_progress(status_counts: dict[str, int], total: int) -> dict[str, Any]:
    finished = sum(
        status_counts.get(status, 0)
        for status in ("auto_merged", "manual_merged", "new_created", "completed")
    )
    needs_review = status_counts.get("needs_review", 0)
    failed = status_counts.get("failed", 0) + status_counts.get("dead_letter", 0)
    active = status_counts.get("pending", 0) + status_counts.get("processing", 0)
    done = finished + needs_review + failed
    return {
        "total": total,
        "active": active,
        "finished": finished,
        "needs_review": needs_review,
        "failed": failed,
        "done": done,
        "percent": round((done / total) * 100, 1) if total else 100.0,
    }


def group_attribute_rows(rows: list[dict[str, Any]]) -> dict[int, list[StoredAttribute]]:
    grouped: dict[int, dict[int, StoredAttribute]] = {}
    for row in rows:
        entity_id = int(row["entity_id"])
        attribute_id = int(row["attribute_id"])
        entity_attributes = grouped.setdefault(entity_id, {})
        attribute = entity_attributes.get(attribute_id)
        if attribute is None:
            attribute = StoredAttribute(
                id=attribute_id,
                attr_type=str(row["attr_type"]),
                value_text=str(row["value_text"] or ""),
                confidence=float(row.get("confidence") or 0.7),
            )
            entity_attributes[attribute_id] = attribute
        if row.get("evidence_text"):
            attribute.evidence.append(
                StoredAttributeEvidence(
                    evidence_text=str(row.get("evidence_text") or ""),
                    note_id=str(row.get("note_id") or ""),
                    block_id=str(row.get("block_id") or ""),
                    block_index=int(row.get("block_index") if row.get("block_index") is not None else -1),
                )
            )
    return {entity_id: list(attributes.values()) for entity_id, attributes in grouped.items()}


def build_summary(incoming: IncomingEntity) -> str:
    definitions = [item.value_text for item in incoming.attributes if item.attr_type == "definition"]
    if definitions:
        return definitions[0]
    if incoming.identity_description:
        return incoming.identity_description[:500]
    return incoming.description_for_match[:500]


def ensure_entity_table_columns(cursor: Any) -> None:
    cursor.execute("SHOW COLUMNS FROM evorag_entities")
    existing_columns = {str(row["Field"]) for row in cursor.fetchall()}
    scope_columns = {
        "workspace_id": "ALTER TABLE evorag_entities ADD COLUMN workspace_id VARCHAR(128) NOT NULL DEFAULT 'local' AFTER id",
        "project_id": "ALTER TABLE evorag_entities ADD COLUMN project_id VARCHAR(128) NOT NULL DEFAULT 'evorag' AFTER workspace_id",
        "collection_id": "ALTER TABLE evorag_entities ADD COLUMN collection_id VARCHAR(128) NOT NULL DEFAULT 'default' AFTER project_id",
        "domain": "ALTER TABLE evorag_entities ADD COLUMN domain VARCHAR(128) NOT NULL DEFAULT 'general' AFTER collection_id",
    }
    for column, statement in scope_columns.items():
        if column not in existing_columns:
            cursor.execute(statement)
    if "identity_description" not in existing_columns:
        cursor.execute(
            """
            ALTER TABLE evorag_entities
            ADD COLUMN identity_description TEXT NOT NULL AFTER aliases_json
            """
        )
    ensure_index(cursor, "evorag_entities", "idx_evorag_entities_scope", "workspace_id, project_id, collection_id, domain")
    ensure_index(cursor, "evorag_entities", "idx_evorag_entities_scope_name", "workspace_id, project_id, collection_id, domain, normalized_name")
    ensure_resolution_audit_scope_columns(cursor)
    ensure_ingest_job_columns(cursor)
    ensure_incoming_entity_columns(cursor)
    ensure_review_table_indexes(cursor)


def ensure_resolution_audit_scope_columns(cursor: Any) -> None:
    cursor.execute("SHOW COLUMNS FROM evorag_entity_resolution_audit")
    existing_columns = {str(row["Field"]) for row in cursor.fetchall()}
    scope_columns = {
        "workspace_id": "ALTER TABLE evorag_entity_resolution_audit ADD COLUMN workspace_id VARCHAR(128) NOT NULL DEFAULT 'local' AFTER id",
        "project_id": "ALTER TABLE evorag_entity_resolution_audit ADD COLUMN project_id VARCHAR(128) NOT NULL DEFAULT 'evorag' AFTER workspace_id",
        "collection_id": "ALTER TABLE evorag_entity_resolution_audit ADD COLUMN collection_id VARCHAR(128) NOT NULL DEFAULT 'default' AFTER project_id",
        "domain": "ALTER TABLE evorag_entity_resolution_audit ADD COLUMN domain VARCHAR(128) NOT NULL DEFAULT 'general' AFTER collection_id",
    }
    for column, statement in scope_columns.items():
        if column not in existing_columns:
            cursor.execute(statement)
    ensure_index(cursor, "evorag_entity_resolution_audit", "idx_evorag_resolution_audit_scope", "workspace_id, project_id, collection_id, domain")


def ensure_ingest_job_columns(cursor: Any) -> None:
    cursor.execute("SHOW TABLES LIKE 'evorag_ingest_jobs'")
    if not cursor.fetchone():
        return
    cursor.execute("SHOW COLUMNS FROM evorag_ingest_jobs")
    existing_columns = {str(row["Field"]) for row in cursor.fetchall()}
    columns = {
        "source_note_id": "ALTER TABLE evorag_ingest_jobs ADD COLUMN source_note_id VARCHAR(128) NOT NULL DEFAULT '' AFTER domain",
        "task_name": "ALTER TABLE evorag_ingest_jobs ADD COLUMN task_name VARCHAR(255) NOT NULL DEFAULT '' AFTER source_note_id",
    }
    for column, statement in columns.items():
        if column not in existing_columns:
            cursor.execute(statement)


def ensure_index(cursor: Any, table_name: str, index_name: str, columns_sql: str) -> None:
    cursor.execute("SHOW INDEX FROM {table_name} WHERE Key_name = %s".format(table_name=table_name), (index_name,))
    if not cursor.fetchone():
        cursor.execute(f"ALTER TABLE {table_name} ADD INDEX {index_name} ({columns_sql})")


def ensure_incoming_entity_columns(cursor: Any) -> None:
    cursor.execute("SHOW TABLES LIKE 'evorag_incoming_entities'")
    if not cursor.fetchone():
        return
    cursor.execute("SHOW COLUMNS FROM evorag_incoming_entities")
    existing_columns = {str(row["Field"]) for row in cursor.fetchall()}
    columns = {
        "matched_entity_id": "ALTER TABLE evorag_incoming_entities ADD COLUMN matched_entity_id BIGINT NULL AFTER attempt_count",
        "decision": "ALTER TABLE evorag_incoming_entities ADD COLUMN decision VARCHAR(32) NOT NULL DEFAULT '' AFTER matched_entity_id",
        "decision_score": "ALTER TABLE evorag_incoming_entities ADD COLUMN decision_score DOUBLE NOT NULL DEFAULT 0 AFTER decision",
        "decision_reason": "ALTER TABLE evorag_incoming_entities ADD COLUMN decision_reason TEXT NOT NULL AFTER decision_score",
    }
    for column, statement in columns.items():
        if column not in existing_columns:
            cursor.execute(statement)


def ensure_review_table_indexes(cursor: Any) -> None:
    cursor.execute("SHOW TABLES LIKE 'evorag_entity_review_tasks'")
    if not cursor.fetchone():
        return
    ensure_index(cursor, "evorag_entity_review_tasks", "idx_evorag_review_incoming_status", "incoming_entity_id, status")
    cursor.execute("SHOW INDEX FROM evorag_entity_review_tasks WHERE Key_name = 'uk_evorag_review_incoming_active'")
    if cursor.fetchone():
        try:
            cursor.execute("ALTER TABLE evorag_entity_review_tasks DROP INDEX uk_evorag_review_incoming_active")
        except Exception as exc:
            if not is_mysql_foreign_key_index_drop_error(exc):
                raise


def is_mysql_foreign_key_index_drop_error(exc: Exception) -> bool:
    args = getattr(exc, "args", ())
    return bool(args and int(args[0]) == 1553)


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


def incoming_entity_key(incoming: IncomingEntity) -> str:
    return f"{normalize_name(incoming.name)}::{normalize_name(incoming.entity_type)}"


