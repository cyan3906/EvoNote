import asyncio

from app.EvoRAG.entity_store.models import EntityScope, StoredEntity
from app.EvoRAG.models import RetrievedEntity, StoredAttribute
from app.EvoRAG.services.graph_generation import EvoRAGGraphGenerationService


def scope() -> EntityScope:
    return EntityScope(workspace_id="local", project_id="evorag", collection_id="default", domain="general")


def retrieved(entity_id: int, name: str, *, attributes: list[StoredAttribute] | None = None) -> RetrievedEntity:
    return RetrievedEntity(
        id=entity_id,
        canonical_name=name,
        normalized_name=name.lower(),
        entity_type="concept",
        identity_description=f"{name} identity",
        summary=f"{name} summary",
        score=1.0,
        rank=entity_id,
        source="test",
        attributes=attributes or [],
    )


def stored(entity_id: int, name: str) -> StoredEntity:
    return StoredEntity(
        id=entity_id,
        canonical_name=name,
        normalized_name=name.lower(),
        entity_type="concept",
        scope=scope(),
        identity_description=f"{name} identity",
        summary=f"{name} summary",
    )


class FakeRetriever:
    async def retrieve(self, query: str, *, top_k: int | None = None):
        assert query == "MVCC"
        assert top_k == 3
        return [
            retrieved(1, "MVCC", attributes=[StoredAttribute(attr_type="mechanism", value_text="MVCC uses Undo Log.", confidence=0.9)]),
            retrieved(2, "Undo Log"),
            retrieved(3, "ReadView"),
        ], ["retriever warning"]


class FakeRepository:
    def __init__(self) -> None:
        self.entities = {
            1: stored(1, "MVCC"),
            2: stored(2, "Undo Log"),
            3: stored(3, "ReadView"),
        }

    def get_entity(self, entity_id: int):
        return self.entities.get(entity_id)

    def list_attributes_for_entities(self, entity_ids: list[int]):
        return {
            1: [StoredAttribute(attr_type="mechanism", value_text="MVCC uses Undo Log and ReadView.", confidence=0.9)],
            2: [StoredAttribute(attr_type="definition", value_text="Undo Log stores old versions.", confidence=0.8)],
            3: [StoredAttribute(attr_type="definition", value_text="ReadView is a snapshot.", confidence=0.8)],
        }


def test_graph_candidates_returns_retrieved_entities_and_dependency_graph() -> None:
    service = EvoRAGGraphGenerationService(retriever=FakeRetriever(), repository=FakeRepository(), scope=scope())

    payload = asyncio.run(service.candidates("MVCC", top_k=3))

    assert payload["root_entity"]["canonical_name"] == "MVCC"
    assert [entity["canonical_name"] for entity in payload["candidates"]] == ["MVCC", "Undo Log", "ReadView"]
    assert payload["default_selected_entity_ids"] == [1, 2, 3]
    assert payload["warnings"] == ["retriever warning"]
    assert payload["graph"]["semantic_edges"][0]["source_id"] == 1
    assert payload["graph"]["semantic_edges"][0]["target_id"] == 2


def test_graph_generate_uses_selected_entities_to_build_graph_outline_and_article() -> None:
    service = EvoRAGGraphGenerationService(retriever=FakeRetriever(), repository=FakeRepository(), scope=scope())

    payload = asyncio.run(service.generate(root_entity_id=1, selected_entity_ids=[3, 2]))

    assert [entity["canonical_name"] for entity in payload["entities"]] == ["MVCC", "ReadView", "Undo Log"]
    assert {edge["target_id"] for edge in payload["graph"]["semantic_edges"]} == {2, 3}
    assert payload["outline"][0]["title"] == "MVCC"
    assert "# MVCC" in payload["article"]
    assert "MVCC -> Undo Log" in payload["article"]
