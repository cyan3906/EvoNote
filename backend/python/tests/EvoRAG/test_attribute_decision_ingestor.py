import asyncio

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.models import (
    AttributeDecision,
    AttributeDecisionApplyResult,
    AttributeRetrievalResult,
    EntityAttributeInput,
    EntityResolutionDecision,
    EntityScope,
    EntityUpsertResult,
    IncomingEntity,
    StoredEntity,
    StoredEntityAttribute,
)


def scope() -> EntityScope:
    return EntityScope(workspace_id="local", project_id="evorag", collection_id="default", domain="general")


def matched_entity() -> StoredEntity:
    return StoredEntity(id=3, canonical_name="Redis 分布式锁", normalized_name="redis分布式锁", entity_type="concept", scope=scope())


def incoming_entity() -> IncomingEntity:
    return IncomingEntity(
        name="Redis 锁",
        normalized_name="redis锁",
        entity_type="concept",
        scope=scope(),
        attributes=[EntityAttributeInput(attr_type="mechanism", value_text="SET NX EX 获取锁", evidence="block evidence")],
    )


class FakeRepository:
    def __init__(self) -> None:
        self.upsert_calls = []
        self.applied_decisions = []
        self.requested_attribute_ids = []
        self.audit_calls = []

    def upsert_entity(self, incoming: IncomingEntity, matched_entity_id: int | None = None) -> EntityUpsertResult:
        self.upsert_calls.append((incoming, matched_entity_id))
        assert incoming.attributes == []
        return EntityUpsertResult(entity_id=matched_entity_id or 99, canonical_name=incoming.name, created=False, attribute_count=0, evidence_count=0)

    def apply_attribute_decisions(self, entity_id: int, decisions: list[AttributeDecision]) -> AttributeDecisionApplyResult:
        self.applied_decisions.append((entity_id, decisions))
        return AttributeDecisionApplyResult(attribute_count=1, evidence_count=1, conflict_count=0, changed_attribute_ids=[10])

    def get_entity(self, entity_id: int) -> StoredEntity:
        return matched_entity()

    def upsert_aliases_for_entity(self, *args, **kwargs) -> None:
        return None

    def get_active_attributes_by_ids(self, attribute_ids: list[int]) -> list[StoredEntityAttribute]:
        self.requested_attribute_ids.append(list(attribute_ids))
        return [
            StoredEntityAttribute(
                id=10,
                entity_id=3,
                scope=scope(),
                attr_type="mechanism",
                value_text="SET NX EX 获取锁",
                value_fingerprint="fp",
                confidence=0.9,
                status="active",
            )
        ]

    def record_resolution_audit(self, *args, **kwargs) -> None:
        self.audit_calls.append((args, kwargs))


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    async def retrieve(self, entity: StoredEntity, incoming_attributes: list[EntityAttributeInput]):
        self.calls.append((entity, incoming_attributes))
        return [AttributeRetrievalResult(input_index=0, attr_type="mechanism", value_text="SET NX EX 获取锁", mode="full_scan", group_size=0)]


class FakeDecider:
    def __init__(self) -> None:
        self.calls = []

    def decide(self, incoming_attributes, retrieval_results):
        self.calls.append((incoming_attributes, retrieval_results))
        return [AttributeDecision(input_index=0, action="add", incoming_attribute=incoming_attributes[0], confidence=0.8, reason="no candidate")]


class FakeEmbedding:
    def __init__(self) -> None:
        self.calls = []

    async def embed_texts(self, texts: list[str]):
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeEntityIndex:
    def __init__(self) -> None:
        self.entities = []

    def upsert_entity(self, entity: StoredEntity) -> None:
        self.entities.append(entity)


class FakeAttributeIndex:
    def __init__(self) -> None:
        self.es_records = []
        self.milvus_records = []

    def upsert_elasticsearch_attributes(self, records):
        self.es_records.append(records)

    def upsert_milvus_attributes(self, records, vectors):
        self.milvus_records.append((records, vectors))


def test_ingestor_runs_attribute_decisions_before_persisting_matched_entity_attributes() -> None:
    repository = FakeRepository()
    retriever = FakeRetriever()
    decider = FakeDecider()
    embedding = FakeEmbedding()
    entity_index = FakeEntityIndex()
    attribute_index = FakeAttributeIndex()
    ingestor = EntityIngestor(
        repository=repository,
        config=EvoRAGSettings(_env_file=None),
        attribute_retriever=retriever,
        attribute_decider=decider,
        embedding_client=embedding,
        hybrid_index=entity_index,
        attribute_index=attribute_index,
    )
    decision = EntityResolutionDecision(
        incoming=incoming_entity(),
        decision="matched",
        matched_entity=matched_entity(),
        score=0.95,
        reason="matched by resolver",
    )

    result = asyncio.run(ingestor.apply_decisions([decision]))[0]

    assert retriever.calls[0][0].id == 3
    assert decider.calls[0][0] == decision.incoming.attributes
    assert repository.upsert_calls[0][0].attributes == []
    assert repository.applied_decisions[0][0] == 3
    assert result.attribute_count == 1
    assert result.evidence_count == 1
    assert result.changed_attribute_ids == [10]
    assert repository.requested_attribute_ids == [[10]]
    assert attribute_index.es_records[0][0].id == 10
    assert attribute_index.milvus_records[0][1] == [[0.1, 0.2, 0.3]]