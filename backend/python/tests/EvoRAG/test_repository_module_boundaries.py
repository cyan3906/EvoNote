from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.entity_store.models import EntityScope
from app.EvoRAG.entity_store.repository import (
    MySQLEntityRepository,
    build_scope_where as legacy_build_scope_where,
    ensure_review_table_indexes as legacy_ensure_review_table_indexes,
    row_to_entity as legacy_row_to_entity,
)
from app.EvoRAG.entity_store.repository_attributes import AttributeDecisionRepositoryMixin
from app.EvoRAG.entity_store.repository_helpers import build_scope_where
from app.EvoRAG.entity_store.repository_ingest import IngestJobRepositoryMixin
from app.EvoRAG.entity_store.repository_mappers import row_to_entity
from app.EvoRAG.entity_store.repository_schema import ensure_review_table_indexes


def test_repository_keeps_legacy_helper_exports_after_module_split() -> None:
    assert legacy_build_scope_where is build_scope_where
    assert legacy_row_to_entity is row_to_entity
    assert legacy_ensure_review_table_indexes is ensure_review_table_indexes


def test_repository_composes_ingest_and_attribute_mixins() -> None:
    assert issubclass(MySQLEntityRepository, IngestJobRepositoryMixin)
    assert issubclass(MySQLEntityRepository, AttributeDecisionRepositoryMixin)
    assert MySQLEntityRepository.create_ingest_job is IngestJobRepositoryMixin.create_ingest_job
    assert MySQLEntityRepository.apply_attribute_decisions is AttributeDecisionRepositoryMixin.apply_attribute_decisions


def test_repository_helpers_preserve_scope_filter_contract() -> None:
    scope = EntityScope(workspace_id="ws", project_id="proj", collection_id="col", domain="general")

    where_sql, params = build_scope_where(scope)

    assert where_sql == "AND workspace_id = %s AND project_id = %s AND collection_id = %s AND domain = %s"
    assert params == ("ws", "proj", "col", "general")


def test_repository_mapper_preserves_entity_shape() -> None:
    entity = row_to_entity(
        {
            "id": 7,
            "workspace_id": "ws",
            "project_id": "proj",
            "collection_id": "col",
            "domain": "general",
            "canonical_name": "MVCC",
            "normalized_name": "mvcc",
            "entity_type": "concept",
            "aliases_json": '["多版本并发控制"]',
            "identity_description": "数据库并发控制机制",
            "summary": "减少读写阻塞",
            "description_for_match": "name: MVCC",
            "embedding_json": "[0.1, 0.2]",
        }
    )

    assert entity.id == 7
    assert entity.scope.workspace_id == "ws"
    assert entity.aliases == ["多版本并发控制"]
    assert entity.embedding == [0.1, 0.2]


def test_repository_helper_uses_evorag_config_defaults() -> None:
    scope = EntityScope(
        workspace_id=EvoRAGSettings(_env_file=None).default_workspace_id,
        project_id="evorag",
        collection_id="default",
        domain="general",
    )

    assert build_scope_where(scope)[1] == ("local", "evorag", "default", "general")
