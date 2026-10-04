from typing import Any


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
    ensure_attribute_decision_tables(cursor)


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


def ensure_attribute_decision_tables(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evorag_entity_attribute_decision_audit (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            entity_id BIGINT NOT NULL,
            input_index INT NOT NULL DEFAULT -1,
            action VARCHAR(32) NOT NULL,
            attr_type VARCHAR(32) NOT NULL,
            incoming_value_text TEXT NOT NULL,
            target_attribute_id BIGINT NULL,
            changed_attribute_id BIGINT NULL,
            new_value_text TEXT NOT NULL,
            confidence DOUBLE NOT NULL DEFAULT 0,
            reason TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_evorag_attr_decision_entity (entity_id),
            INDEX idx_evorag_attr_decision_action (action)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evorag_entity_attribute_conflicts (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            entity_id BIGINT NOT NULL,
            attribute_id BIGINT NOT NULL,
            attr_type VARCHAR(32) NOT NULL,
            incoming_value_text TEXT NOT NULL,
            incoming_evidence TEXT NOT NULL,
            reason TEXT NOT NULL,
            confidence DOUBLE NOT NULL DEFAULT 0,
            status VARCHAR(32) NOT NULL DEFAULT 'pending',
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_evorag_attr_conflict_entity (entity_id),
            INDEX idx_evorag_attr_conflict_attribute (attribute_id),
            INDEX idx_evorag_attr_conflict_status (status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evorag_entity_attribute_events (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            entity_id BIGINT NOT NULL,
            attribute_id BIGINT NOT NULL,
            attr_type VARCHAR(32) NOT NULL,
            action VARCHAR(32) NOT NULL,
            review_status VARCHAR(32) NOT NULL DEFAULT 'pending',
            before_json JSON NULL,
            after_json JSON NULL,
            incoming_json JSON NOT NULL,
            candidate_json JSON NULL,
            diff_json JSON NOT NULL,
            decision_reason TEXT NOT NULL,
            confidence DOUBLE NOT NULL DEFAULT 0,
            source VARCHAR(64) NOT NULL DEFAULT 'auto',
            note_id VARCHAR(128) NOT NULL DEFAULT '',
            block_id VARCHAR(128) NOT NULL DEFAULT '',
            block_index INT NOT NULL DEFAULT -1,
            corrected_by_event_id BIGINT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_evorag_attr_event_entity (entity_id),
            INDEX idx_evorag_attr_event_attribute (attribute_id),
            INDEX idx_evorag_attr_event_action (action),
            INDEX idx_evorag_attr_event_review (review_status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
    )
