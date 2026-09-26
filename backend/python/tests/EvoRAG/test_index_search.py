import asyncio
from types import SimpleNamespace

from app.EvoRAG.entity_store.models import CandidateEntity, EntityScope, StoredEntity
from app.EvoRAG.models import StoredAttribute
from app.EvoRAG.services.retriever import EvoRAGRetriever


def run(coro):
    return asyncio.run(coro)


def config() -> SimpleNamespace:
    return SimpleNamespace(
        default_workspace_id="local",
        default_project_id="evorag",
        default_collection_id="default",
        default_domain="general",
        entity_resolution_top_k=3,
        entity_resolution_rrf_k=60,
    )


def stored(entity_id: int, name: str) -> StoredEntity:
    return StoredEntity(
        id=entity_id,
        canonical_name=name,
        normalized_name=name.lower(),
        entity_type="concept",
        scope=EntityScope(),
        aliases=[f"{name} alias"],
        identity_description=f"{name} identity",
        summary=f"{name} summary",
        description_for_match=f"{name} match text",
    )


class FakeRepository:
    def __init__(self) -> None:
        self.init_schema_called = False
        self.hydrate_calls: list[list[int]] = []
        self.entities = {
            1: stored(1, "ES Entity"),
            2: stored(2, "Milvus Entity"),
        }

    def init_schema(self) -> None:
        self.init_schema_called = True

    def get_entity(self, entity_id: int) -> StoredEntity | None:
        return self.entities.get(entity_id)

    def list_attributes_for_entities(self, entity_ids: list[int]) -> dict[int, list[StoredAttribute]]:
        return {
            entity_id: [
                StoredAttribute(
                    attr_type="definition",
                    value_text=f"Definition for entity {entity_id}",
                    confidence=0.9,
                )
            ]
            for entity_id in entity_ids
        }

    def hydrate_entities_for_candidates(self, entity_ids: list[int]) -> tuple[dict[int, StoredEntity], dict[int, list[StoredAttribute]]]:
        self.hydrate_calls.append(entity_ids)
        unique_ids: list[int] = []
        for entity_id in entity_ids:
            if entity_id not in unique_ids:
                unique_ids.append(entity_id)
        return (
            {entity_id: self.entities[entity_id] for entity_id in unique_ids if entity_id in self.entities},
            self.list_attributes_for_entities(unique_ids),
        )


class FakeEmbeddingClient:
    async def embed_text(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeHybridIndex:
    def __init__(self) -> None:
        self.elasticsearch_ensured = False
        self.milvus_ensured = False

    def ensure_elasticsearch_index(self) -> None:
        self.elasticsearch_ensured = True

    def ensure_milvus_collection(self) -> None:
        self.milvus_ensured = True

    def search_elasticsearch(self, incoming, *, top_k: int) -> list[CandidateEntity]:
        return [CandidateEntity(entity=stored(1, "ES Entity"), score=1.0, rank=1, source="elasticsearch", es_score=1.0)]

    def search_milvus(self, vector: list[float], *, top_k: int, scope: EntityScope | None = None) -> list[CandidateEntity]:
        assert vector == [0.1, 0.2, 0.3]
        return [CandidateEntity(entity=stored(2, "Milvus Entity"), score=0.8, rank=1, source="milvus", vector_score=0.8)]

    def backend_status(self) -> dict[str, object]:
        return {
            "core": {
                "initialized": True,
                "elasticsearch": {"available": True},
                "milvus": {"available": True},
            },
            "evorag": {
                "mysql": {"available": True, "initialized": True},
                "elasticsearch": {"uses_core_client": True},
                "milvus": {"uses_core_client": True},
            },
        }


def test_index_search_returns_es_milvus_and_fused_results_with_attributes() -> None:
    repository = FakeRepository()
    hybrid_index = FakeHybridIndex()
    retriever = EvoRAGRetriever(
        repository=repository,
        hybrid_index=hybrid_index,
        embedding_client=FakeEmbeddingClient(),
        config=config(),
    )

    result = run(retriever.search_indexes("entity query", top_k=3))

    assert not repository.init_schema_called
    assert repository.hydrate_calls == [[1, 2, 1, 2]]
    assert hybrid_index.elasticsearch_ensured
    assert hybrid_index.milvus_ensured
    assert result.query == "entity query"
    assert [entity.canonical_name for entity in result.elasticsearch_results] == ["ES Entity"]
    assert [entity.canonical_name for entity in result.milvus_results] == ["Milvus Entity"]
    assert [entity.canonical_name for entity in result.fused_results] == ["ES Entity", "Milvus Entity"]
    assert result.elasticsearch_results[0].attributes[0].value_text == "Definition for entity 1"
    assert result.milvus_results[0].attributes[0].value_text == "Definition for entity 2"
    assert result.backend_status["evorag"]["mysql"]["available"] is True
    assert result.backend_status["evorag"]["elasticsearch"]["uses_core_client"] is True
    assert result.backend_status["evorag"]["milvus"]["uses_core_client"] is True
    assert result.timings["total_ms"] >= 0
    assert result.timings["elasticsearch_search_ms"] >= 0
    assert result.timings["milvus_search_ms"] >= 0
    assert result.timings["fusion_ms"] >= 0
    assert result.warnings == []
