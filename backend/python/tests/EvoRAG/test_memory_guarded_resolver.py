import asyncio

from app.EvoRAG.entity_store.models import CandidateEntity, EntityRelationMemoryRecord, EntityScope, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store import resolver as resolver_module
from app.EvoRAG.entity_store.resolver import EntityResolver


def test_resolver_uses_direct_relation_memory_allow_before_hybrid() -> None:
    candidate = stored_entity()
    relation_memory = FakeRelationMemory(direct_relation=relation(candidate=candidate))
    hybrid_index = FakeHybridIndex()
    resolver = EntityResolver(
        [candidate],
        llm_client=FakeLLM([]),
        hybrid_index=hybrid_index,
        relation_memory=relation_memory,
    )

    decision = asyncio.run(resolver.resolve_one(incoming_entity()))

    assert decision.decision == "matched"
    assert decision.matched_entity == candidate
    assert decision.reason == "relation memory direct allow"
    assert decision.candidates[0].source == "relation_memory"
    assert hybrid_index.calls == []
    assert relation_memory.list_calls == 0


def test_resolver_uses_mysql_relation_memory_judge_to_match() -> None:
    candidate = stored_entity()
    relation_memory = FakeRelationMemory(relations=[relation(candidate=candidate)])
    llm = FakeLLM(
        [
            {"decision": "entity", "confidence": 0.9, "reason": "是实体"},
            {"decision": "matched", "matched_entity_id": candidate.id, "confidence": 0.88, "reason": "历史白名单确认"},
        ]
    )
    resolver = EntityResolver(
        [candidate],
        llm_client=llm,
        hybrid_index=FakeHybridIndex(),
        relation_memory=relation_memory,
    )

    decision = asyncio.run(resolver.resolve_one(incoming_entity()))

    assert decision.decision == "matched"
    assert decision.matched_entity == candidate
    assert decision.score == 0.88
    assert decision.candidates[0].source == "relation_memory"
    assert llm.calls[1]["user_payload"]["allow_relations"][0]["candidate_entity"]["id"] == candidate.id


def test_resolver_lazy_loads_llm_client_for_actual_ingest_flow(monkeypatch) -> None:
    candidate = stored_entity()
    candidate.aliases = []
    relation_memory = FakeRelationMemory(relations=[relation(candidate=candidate)])
    llm = FakeLLM(
        [
            {"decision": "entity", "confidence": 0.9, "reason": "是实体"},
            {"decision": "matched", "matched_entity_id": candidate.id, "confidence": 0.88, "reason": "历史白名单确认"},
        ]
    )
    monkeypatch.setattr(resolver_module, "EvoRAGLLMClient", lambda config: llm)
    resolver = EntityResolver(
        [candidate],
        hybrid_index=FakeHybridIndex(),
        relation_memory=relation_memory,
    )

    decision = asyncio.run(resolver.resolve_one(incoming_entity()))

    assert decision.decision == "matched"
    assert decision.matched_entity == candidate
    assert len(llm.calls) == 2


def test_resolver_turns_relation_guard_non_merge_into_new_with_reject_experience_metadata() -> None:
    candidate = stored_entity()
    relation_memory = FakeRelationMemory(relations=[relation(decision="reject", relation_type="manual_reject", candidate=candidate)])
    llm = FakeLLM(
        [
            {"decision": "entity", "confidence": 0.9, "reason": "是实体"},
            {"decision": "attribute_of_entity", "matched_entity_id": candidate.id, "confidence": 0.83, "reason": "只是候选实体属性"},
        ]
    )
    resolver = EntityResolver(
        [candidate],
        llm_client=llm,
        hybrid_index=FakeHybridIndex(),
        relation_memory=relation_memory,
    )

    decision = asyncio.run(resolver.resolve_one(incoming_entity()))

    assert decision.decision == "new"
    assert decision.matched_entity is None
    assert decision.candidates[0].entity == candidate
    assert decision.record_experience is True
    assert decision.experience_decision == "reject"
    assert decision.experience_relation_type == "attribute_of_entity"
    assert decision.experience_source == "llm_guard"
    assert decision.experience_confidence == 0.83


def test_resolver_uses_final_llm_judge_after_hybrid_fallback() -> None:
    candidate = stored_entity()
    candidate.aliases = []
    llm = FakeLLM(
        [
            {"decision": "entity", "confidence": 0.9, "reason": "是实体"},
            {"decision": "matched", "matched_entity_id": candidate.id, "confidence": 0.76, "reason": "候选描述一致"},
        ]
    )
    hybrid_index = FakeHybridIndex([CandidateEntity(entity=candidate, score=0.72, rank=1, source="hybrid")])
    resolver = EntityResolver(
        [candidate],
        llm_client=llm,
        hybrid_index=hybrid_index,
        relation_memory=FakeRelationMemory(),
    )

    decision = asyncio.run(resolver.resolve_one(incoming_entity()))

    assert decision.decision == "matched"
    assert decision.matched_entity == candidate
    assert llm.calls[1]["user_payload"]["candidates"][0]["id"] == candidate.id
    assert decision.record_experience is True
    assert decision.experience_decision == "allow"
    assert decision.experience_relation_type == "llm_judge_match"
    assert decision.experience_source == "llm_judge"


class FakeRelationMemory:
    def __init__(
        self,
        *,
        direct_relation: EntityRelationMemoryRecord | None = None,
        relations: list[EntityRelationMemoryRecord] | None = None,
    ) -> None:
        self.direct_relation = direct_relation
        self.relations = relations or []
        self.list_calls = 0

    def get_direct_relation(self, incoming: IncomingEntity) -> EntityRelationMemoryRecord | None:
        return self.direct_relation

    def list_top_relations(self, incoming: IncomingEntity, *, limit: int = 30) -> list[EntityRelationMemoryRecord]:
        self.list_calls += 1
        assert limit == 30
        return self.relations


class FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeHybridIndex:
    def __init__(self, candidates: list[CandidateEntity] | None = None) -> None:
        self.candidates = candidates or []
        self.calls: list[dict] = []

    def search(self, incoming: IncomingEntity, top_k: int) -> list[CandidateEntity]:
        self.calls.append({"incoming": incoming, "top_k": top_k})
        return self.candidates


def relation(
    *,
    decision: str = "allow",
    relation_type: str = "manual_match",
    candidate: StoredEntity | None = None,
) -> EntityRelationMemoryRecord:
    return EntityRelationMemoryRecord(
        id=11,
        decision=decision,
        relation_type=relation_type,
        candidate=candidate or stored_entity(),
        confidence=0.91,
        hit_count=3,
        source="manual",
        reason="历史经验",
    )


def incoming_entity() -> IncomingEntity:
    return IncomingEntity(
        name="多版本并发控制",
        normalized_name="多版本并发控制",
        entity_type="concept",
        scope=scope(),
        aliases=[],
        identity_description="减少读写阻塞的并发控制机制。",
        description_for_match="数据库并发控制机制。",
    )


def stored_entity() -> StoredEntity:
    return StoredEntity(
        id=7,
        canonical_name="MVCC",
        normalized_name="mvcc",
        entity_type="concept",
        scope=scope(),
        aliases=["多版本并发控制"],
        identity_description="数据库多版本并发控制机制。",
        summary="通过多版本降低读写阻塞。",
        description_for_match="数据库并发控制机制。",
    )


def scope() -> EntityScope:
    return EntityScope(workspace_id="eval", project_id="entity-resolution", collection_id="tiny", domain="computer-science")
