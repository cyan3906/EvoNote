import json
from pathlib import Path
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.alias_cache import EntityAliasCache, scope_hash
from app.EvoRAG.entity_store.models import CandidateEntity, EntityAttributeInput, EntityRelationMemoryRecord, EntityScope, EntityUpsertResult, IncomingEntity, IncomingEntityTask, StoredEntity, StoredEntityAttribute
from app.EvoRAG.entity_store.normalizer import normalize_name, text_fingerprint
from app.EvoRAG.entity_store.repository_attributes import AttributeDecisionRepositoryMixin
from app.EvoRAG.entity_store.repository_ingest import IngestJobRepositoryMixin
from app.EvoRAG.models import StoredAttribute
from app.core.evorag_database import connect_evorag_mysql


class MySQLEntityRepository(IngestJobRepositoryMixin, AttributeDecisionRepositoryMixin):
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

    def count_active_attributes_by_type(self, entity_id: int, attr_types: list[str]) -> dict[str, int]:
        unique_types = unique_strings(attr_types, sort=False)
        if not unique_types:
            return {}

        placeholders = ", ".join(["%s"] * len(unique_types))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT attr_type, COUNT(*) AS attribute_count
                    FROM evorag_entity_attributes
                    WHERE entity_id = %s
                      AND status = 'active'
                      AND attr_type IN ({placeholders})
                    GROUP BY attr_type
                    """,
                    (int(entity_id), *unique_types),
                )
                rows = cursor.fetchall()
        counts = {attr_type: 0 for attr_type in unique_types}
        counts.update({str(row["attr_type"]): int(row["attribute_count"]) for row in rows})
        return counts

    def find_exact_active_attributes(
        self,
        entity_id: int,
        fingerprints_by_type: dict[str, set[str]],
    ) -> dict[tuple[str, str], StoredEntityAttribute]:
        attr_types = sorted(str(attr_type) for attr_type in fingerprints_by_type if fingerprints_by_type[attr_type])
        fingerprints = sorted({str(fp) for values in fingerprints_by_type.values() for fp in values if fp})
        if not attr_types or not fingerprints:
            return {}

        type_placeholders = ", ".join(["%s"] * len(attr_types))
        fingerprint_placeholders = ", ".join(["%s"] * len(fingerprints))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        a.id AS attribute_id,
                        a.entity_id,
                        e.workspace_id,
                        e.project_id,
                        e.collection_id,
                        e.domain,
                        a.attr_type,
                        a.value_text,
                        a.value_fingerprint,
                        a.confidence,
                        a.status
                    FROM evorag_entity_attributes a
                    INNER JOIN evorag_entities e ON e.id = a.entity_id
                    WHERE a.entity_id = %s
                      AND a.status = 'active'
                      AND a.attr_type IN ({type_placeholders})
                      AND a.value_fingerprint IN ({fingerprint_placeholders})
                    ORDER BY a.id
                    """,
                    (int(entity_id), *attr_types, *fingerprints),
                )
                rows = cursor.fetchall()

        requested_pairs = {
            (str(attr_type), str(fingerprint))
            for attr_type, values in fingerprints_by_type.items()
            for fingerprint in values
        }
        matches: dict[tuple[str, str], StoredEntityAttribute] = {}
        for row in rows:
            pair = (str(row["attr_type"]), str(row["value_fingerprint"]))
            if pair in requested_pairs and pair not in matches:
                matches[pair] = row_to_entity_attribute(row)
        return matches

    def list_active_attributes_for_types(
        self,
        entity_id: int,
        attr_types: list[str],
    ) -> list[StoredEntityAttribute]:
        unique_types = unique_strings(attr_types, sort=True)
        if not unique_types:
            return []

        placeholders = ", ".join(["%s"] * len(unique_types))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        a.id AS attribute_id,
                        a.entity_id,
                        e.workspace_id,
                        e.project_id,
                        e.collection_id,
                        e.domain,
                        a.attr_type,
                        a.value_text,
                        a.value_fingerprint,
                        a.confidence,
                        a.status
                    FROM evorag_entity_attributes a
                    INNER JOIN evorag_entities e ON e.id = a.entity_id
                    WHERE a.entity_id = %s
                      AND a.status = 'active'
                      AND a.attr_type IN ({placeholders})
                    ORDER BY a.attr_type, a.id
                    """,
                    (int(entity_id), *unique_types),
                )
                rows = cursor.fetchall()
        return [row_to_entity_attribute(row) for row in rows]

    def get_active_attributes_by_ids(self, attribute_ids: list[int]) -> list[StoredEntityAttribute]:
        unique_ids = sorted(unique_ints(attribute_ids))
        if not unique_ids:
            return []

        placeholders = ", ".join(["%s"] * len(unique_ids))
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        a.id AS attribute_id,
                        a.entity_id,
                        e.workspace_id,
                        e.project_id,
                        e.collection_id,
                        e.domain,
                        a.attr_type,
                        a.value_text,
                        a.value_fingerprint,
                        a.confidence,
                        a.status
                    FROM evorag_entity_attributes a
                    INNER JOIN evorag_entities e ON e.id = a.entity_id
                    WHERE a.id IN ({placeholders})
                      AND a.status = 'active'
                    ORDER BY a.id
                    """,
                    tuple(unique_ids),
                )
                rows = cursor.fetchall()
        return [row_to_entity_attribute(row) for row in rows]

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
                            domain = %s
                        WHERE id = %s
                        """,
                        (
                            json.dumps(incoming.aliases, ensure_ascii=False),
                            incoming.scope.workspace_id,
                            incoming.scope.project_id,
                            incoming.scope.collection_id,
                            incoming.scope.domain,
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

    def connect(self):
        return connect_evorag_mysql(self.config)


from app.EvoRAG.entity_store.repository_helpers import (
    build_scope_where,
    incoming_entity_key,
    scope_from_config,
    unique_ints,
    unique_strings,
    update_ingest_job_status,
)
from app.EvoRAG.entity_store.repository_mappers import (
    attribute_event_diff,
    attribute_snapshot,
    build_job_progress,
    build_summary,
    candidate_snapshot,
    event_summary,
    group_attribute_rows,
    incoming_snapshot,
    incoming_status_row_to_dict,
    job_row_to_dict,
    review_row_to_dict,
    review_stats,
    row_to_entity,
    row_to_entity_attribute,
    row_to_incoming_task,
)
from app.EvoRAG.entity_store.repository_schema import (
    ensure_attribute_decision_tables,
    ensure_entity_table_columns,
    ensure_index,
    ensure_ingest_job_columns,
    ensure_incoming_entity_columns,
    ensure_resolution_audit_scope_columns,
    ensure_review_table_indexes,
    is_mysql_foreign_key_index_drop_error,
)
