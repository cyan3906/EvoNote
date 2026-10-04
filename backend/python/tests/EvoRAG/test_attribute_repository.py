from contextlib import contextmanager

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.entity_store.models import (
    AttributeCandidate,
    AttributeRetrievalResult,
    EntityScope,
    StoredEntityAttribute,
)
from app.EvoRAG.entity_store.repository import MySQLEntityRepository


def test_attribute_retrieval_config_defaults_select_adaptive_hybrid_search() -> None:
    config = EvoRAGSettings(_env_file=None)

    assert config.attribute_full_scan_threshold == 20
    assert config.attribute_resolution_top_k == 10
    assert config.attribute_resolution_rrf_k == 60
    assert config.attribute_milvus_max_concurrency == 4
    assert config.attribute_index_backfill_batch_size == 100
    assert config.es_attribute_index == "evorag_entity_attributes"
    assert config.milvus_attribute_collection == "evorag_entity_attributes"


def test_attribute_retrieval_models_preserve_identity_scores_and_warnings() -> None:
    candidate = AttributeCandidate(
        attribute_id=7,
        entity_id=3,
        attr_type="definition",
        value_text="A definition",
        confidence=0.8,
        rank=1,
        source="elasticsearch+milvus",
        es_score=0.75,
        vector_score=0.82,
        fused_score=0.0325,
    )

    result = AttributeRetrievalResult(
        input_index=2,
        attr_type="definition",
        value_text="Incoming definition",
        mode="hybrid",
        group_size=24,
        candidates=[candidate],
        warnings=["milvus timeout"],
    )

    assert result.input_index == 2
    assert result.mode == "hybrid"
    assert result.group_size == 24
    assert result.candidates == [candidate]
    assert result.warnings == ["milvus timeout"]
    assert candidate.es_score == 0.75
    assert candidate.vector_score == 0.82
    assert candidate.fused_score == 0.0325


class ScriptedCursor:
    def __init__(self, results: list[list[dict]]) -> None:
        self.results = list(results)
        self.statements: list[tuple[str, tuple | None]] = []
        self._rows: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, params=None) -> None:
        self.statements.append((query, params))
        self._rows = self.results.pop(0) if self.results else []

    def fetchall(self) -> list[dict]:
        return self._rows


class FakeConnection:
    def __init__(self, cursor: ScriptedCursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self) -> ScriptedCursor:
        return self._cursor


def repository_with_results(results: list[list[dict]]) -> tuple[MySQLEntityRepository, ScriptedCursor]:
    cursor = ScriptedCursor(results)
    repository = MySQLEntityRepository(EvoRAGSettings(_env_file=None))

    @contextmanager
    def connect():
        yield FakeConnection(cursor)

    repository.connect = connect  # type: ignore[method-assign]
    return repository, cursor


def attribute_row(
    attribute_id: int,
    *,
    entity_id: int = 3,
    attr_type: str = "definition",
    value_text: str = "A definition",
    value_fingerprint: str = "fingerprint",
    status: str = "active",
) -> dict:
    return {
        "attribute_id": attribute_id,
        "entity_id": entity_id,
        "workspace_id": "local",
        "project_id": "evorag",
        "collection_id": "default",
        "domain": "general",
        "attr_type": attr_type,
        "value_text": value_text,
        "value_fingerprint": value_fingerprint,
        "confidence": 0.8,
        "status": status,
    }


def test_repository_counts_active_attributes_and_fills_missing_types_with_zero() -> None:
    repository, cursor = repository_with_results(
        [[{"attr_type": "definition", "attribute_count": 19}]]
    )

    counts = repository.count_active_attributes_by_type(
        3, ["definition", "constraints", "definition"]
    )

    assert counts == {"definition": 19, "constraints": 0}
    query, params = cursor.statements[0]
    assert "status = 'active'" in query
    assert params == (3, "definition", "constraints")


def test_repository_exact_lookup_filters_unrequested_type_fingerprint_pairs() -> None:
    repository, cursor = repository_with_results(
        [[
            attribute_row(1, value_fingerprint="fp-a"),
            attribute_row(2, attr_type="constraints", value_fingerprint="fp-b"),
            attribute_row(3, attr_type="constraints", value_fingerprint="fp-a"),
        ]]
    )

    matches = repository.find_exact_active_attributes(
        3,
        {
            "definition": {"fp-a"},
            "constraints": {"fp-b"},
        },
    )

    assert set(matches) == {("definition", "fp-a"), ("constraints", "fp-b")}
    assert matches[("definition", "fp-a")].id == 1
    assert matches[("constraints", "fp-b")].id == 2
    query, params = cursor.statements[0]
    assert "status = 'active'" in query
    assert params == (3, "constraints", "definition", "fp-a", "fp-b")


def test_repository_lists_active_attributes_for_types_in_id_order() -> None:
    repository, cursor = repository_with_results(
        [[
            attribute_row(4, attr_type="constraints"),
            attribute_row(9, attr_type="definition"),
        ]]
    )

    records = repository.list_active_attributes_for_types(
        3, ["definition", "constraints", "definition"]
    )

    assert [record.id for record in records] == [4, 9]
    assert all(isinstance(record, StoredEntityAttribute) for record in records)
    assert records[0].scope == EntityScope()
    query, params = cursor.statements[0]
    assert "status = 'active'" in query
    assert "ORDER BY a.attr_type, a.id" in query
    assert params == (3, "constraints", "definition")


def test_repository_hydrates_unique_active_attribute_ids_in_id_order() -> None:
    repository, cursor = repository_with_results(
        [[attribute_row(2), attribute_row(7, attr_type="constraints")]]
    )

    records = repository.get_active_attributes_by_ids([7, 2, 7])

    assert [record.id for record in records] == [2, 7]
    query, params = cursor.statements[0]
    assert "status = 'active'" in query
    assert "ORDER BY a.id" in query
    assert params == (2, 7)
