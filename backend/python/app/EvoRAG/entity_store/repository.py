import json
from pathlib import Path
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import EntityAttributeInput, EntityScope, EntityUpsertResult, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name, text_fingerprint
from app.EvoRAG.models import StoredAttribute, StoredAttributeEvidence


class MySQLEntityRepository:
    def __init__(self, config: EvoRAGSettings = settings) -> None:
        self.config = config

    def init_schema(self) -> None:
        sql_path = Path(__file__).resolve().parents[1] / "sql" / "entity_schema.sql"
        statements = [statement.strip() for statement in sql_path.read_text(encoding="utf-8").split(";") if statement.strip()]
        with self.connect() as connection:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
                ensure_entity_table_columns(cursor)
            connection.commit()

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
        try:
            import pymysql  # type: ignore
        except Exception as exc:
            raise RuntimeError("pymysql is required for MySQL storage. Install pymysql>=1.1.0.") from exc

        return pymysql.connect(
            host=self.config.mysql_host,
            port=self.config.mysql_port,
            user=self.config.mysql_user,
            password=self.config.mysql_password,
            database=self.config.mysql_database,
            charset=self.config.mysql_charset,
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )


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


def ensure_index(cursor: Any, table_name: str, index_name: str, columns_sql: str) -> None:
    cursor.execute("SHOW INDEX FROM {table_name} WHERE Key_name = %s".format(table_name=table_name), (index_name,))
    if not cursor.fetchone():
        cursor.execute(f"ALTER TABLE {table_name} ADD INDEX {index_name} ({columns_sql})")


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
