from app.EvoRAG.entity_store.repository import ensure_review_table_indexes


class FakeCursor:
    def __init__(self, *, drop_error: Exception | None = None) -> None:
        self.drop_error = drop_error
        self.statements: list[str] = []
        self._last_result = None

    def execute(self, query: str, params=None) -> None:
        self.statements.append(query)
        if query == "SHOW TABLES LIKE 'evorag_entity_review_tasks'":
            self._last_result = {"Tables_in_test": "evorag_entity_review_tasks"}
            return
        if query.startswith("SHOW INDEX FROM evorag_entity_review_tasks") and params == ("idx_evorag_review_incoming_status",):
            self._last_result = None
            return
        if query == "SHOW INDEX FROM evorag_entity_review_tasks WHERE Key_name = 'uk_evorag_review_incoming_active'":
            self._last_result = {"Key_name": "uk_evorag_review_incoming_active"}
            return
        if query == "ALTER TABLE evorag_entity_review_tasks DROP INDEX uk_evorag_review_incoming_active" and self.drop_error:
            raise self.drop_error
        self._last_result = None

    def fetchone(self):
        return self._last_result


class CannotDropForeignKeyIndexError(Exception):
    def __init__(self) -> None:
        super().__init__(1553, "Cannot drop index 'uk_evorag_review_incoming_active': needed in a foreign key constraint")


def test_review_index_migration_creates_replacement_index_before_dropping_old_unique_index() -> None:
    cursor = FakeCursor()

    ensure_review_table_indexes(cursor)

    add_index_position = cursor.statements.index(
        "ALTER TABLE evorag_entity_review_tasks ADD INDEX idx_evorag_review_incoming_status (incoming_entity_id, status)"
    )
    drop_index_position = cursor.statements.index(
        "ALTER TABLE evorag_entity_review_tasks DROP INDEX uk_evorag_review_incoming_active"
    )
    assert add_index_position < drop_index_position


def test_review_index_migration_ignores_mysql_foreign_key_index_drop_error() -> None:
    cursor = FakeCursor(drop_error=CannotDropForeignKeyIndexError())

    ensure_review_table_indexes(cursor)

    assert "ALTER TABLE evorag_entity_review_tasks DROP INDEX uk_evorag_review_incoming_active" in cursor.statements
