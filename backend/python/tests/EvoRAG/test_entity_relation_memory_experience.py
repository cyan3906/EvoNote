import asyncio

from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.models import CandidateEntity, EntityResolutionDecision, EntityScope, EntityUpsertResult, IncomingEntity, IncomingEntityTask, StoredEntity
from app.EvoRAG.entity_store.repository import MySQLEntityRepository


def test_low_score_new_decision_records_reject_experience() -> None:
    repository = FakeExperienceRepository()
    ingestor = EntityIngestor(repository=repository)
    ingestor.hybrid_index = FakeHybridIndex()
    incoming = incoming_entity()
    candidate = stored_entity()
    decision = EntityResolutionDecision(
        incoming=incoming,
        decision="new",
        score=0.49,
        reason="candidate score below direct reject threshold",
        candidates=[CandidateEntity(entity=candidate, score=0.49, rank=1, source="fake")],
    )

    asyncio.run(ingestor.apply_decisions([decision]))

    assert repository.relation_memory_records == [
        {
            "incoming_name": "多版本并发控制",
            "candidate_id": 1,
            "decision": "reject",
            "relation_type": "low_score_direct_reject",
            "confidence": 0.51,
            "source": "auto",
            "reason": "candidate score below direct reject threshold",
        }
    ]


def test_low_score_new_decision_does_not_record_middle_score_experience() -> None:
    repository = FakeExperienceRepository()
    ingestor = EntityIngestor(repository=repository)
    ingestor.hybrid_index = FakeHybridIndex()
    incoming = incoming_entity()
    candidate = stored_entity()
    decision = EntityResolutionDecision(
        incoming=incoming,
        decision="new",
        score=0.55,
        reason="middle score is not in the first supported reuse rules",
        candidates=[CandidateEntity(entity=candidate, score=0.55, rank=1, source="fake")],
    )

    asyncio.run(ingestor.apply_decisions([decision]))

    assert repository.relation_memory_records == []


def test_llm_guard_new_decision_records_reject_experience() -> None:
    repository = FakeExperienceRepository()
    ingestor = EntityIngestor(repository=repository)
    ingestor.hybrid_index = FakeHybridIndex()
    incoming = incoming_entity()
    candidate = stored_entity()
    decision = EntityResolutionDecision(
        incoming=incoming,
        decision="new",
        score=0.83,
        reason="只是候选实体属性",
        candidates=[CandidateEntity(entity=candidate, score=0.83, rank=1, source="relation_memory")],
        record_experience=True,
        experience_decision="reject",
        experience_relation_type="attribute_of_entity",
        experience_source="llm_guard",
        experience_confidence=0.83,
    )

    asyncio.run(ingestor.apply_decisions([decision]))

    assert repository.relation_memory_records == [
        {
            "incoming_name": "多版本并发控制",
            "candidate_id": 1,
            "decision": "reject",
            "relation_type": "attribute_of_entity",
            "confidence": 0.83,
            "source": "llm_guard",
            "reason": "只是候选实体属性",
        }
    ]


def test_llm_judge_matched_decision_records_allow_experience() -> None:
    repository = FakeExperienceRepository()
    ingestor = EntityIngestor(repository=repository)
    ingestor.hybrid_index = FakeHybridIndex()
    incoming = incoming_entity()
    candidate = stored_entity()
    decision = EntityResolutionDecision(
        incoming=incoming,
        decision="matched",
        matched_entity=candidate,
        score=0.76,
        reason="候选描述一致",
        candidates=[CandidateEntity(entity=candidate, score=0.72, rank=1, source="hybrid")],
        record_experience=True,
        experience_decision="allow",
        experience_relation_type="llm_judge_match",
        experience_source="llm_judge",
        experience_confidence=0.76,
    )

    asyncio.run(ingestor.apply_decisions([decision]))

    assert repository.relation_memory_records == [
        {
            "incoming_name": "多版本并发控制",
            "candidate_id": 1,
            "decision": "allow",
            "relation_type": "llm_judge_match",
            "confidence": 0.76,
            "source": "llm_judge",
            "reason": "候选描述一致",
        }
    ]


def test_manual_merge_records_allow_experience() -> None:
    repository = FakeExperienceRepository()
    ingestor = EntityIngestor(repository=repository)
    ingestor.hybrid_index = FakeHybridIndex()

    asyncio.run(ingestor.manual_merge_review_task(7, entity_id=1, reason="人工确认同一实体"))

    assert repository.relation_memory_records == [
        {
            "incoming_name": "多版本并发控制",
            "candidate_id": 1,
            "decision": "allow",
            "relation_type": "manual_match",
            "confidence": 1.0,
            "source": "manual",
            "reason": "人工确认同一实体",
        }
    ]


def test_manual_create_new_records_reject_experience_for_review_candidates() -> None:
    repository = FakeExperienceRepository(review_candidates=[{"id": 1, "canonical_name": "MVCC"}])
    ingestor = EntityIngestor(repository=repository)
    ingestor.hybrid_index = FakeHybridIndex()

    asyncio.run(ingestor.manual_create_new_review_task(7, reason="人工确认不是同一实体"))

    assert repository.relation_memory_records == [
        {
            "incoming_name": "多版本并发控制",
            "candidate_id": 1,
            "decision": "reject",
            "relation_type": "manual_reject",
            "confidence": 1.0,
            "source": "manual",
            "reason": "人工确认不是同一实体",
        }
    ]


def test_repository_relation_memory_upsert_uses_null_entity_id_for_external_candidate() -> None:
    repository = FakeSQLRepository(existing_entity_ids=set())

    repository.upsert_entity_resolution_relation_memory(
        incoming=incoming_entity(),
        candidate=stored_entity(),
        decision="reject",
        relation_type="manual_reject",
        confidence=1.0,
        source="manual",
        reason="eval seed candidate is not stored in mysql entities",
    )

    _, params = repository.cursor.statements[-1]
    assert params[8] is None

def test_repository_relation_memory_upsert_increments_hit_count() -> None:
    repository = FakeSQLRepository()

    repository.upsert_entity_resolution_relation_memory(
        incoming=incoming_entity(),
        candidate=stored_entity(),
        decision="reject",
        relation_type="low_score_direct_reject",
        confidence=0.51,
        source="auto",
        reason="score below threshold",
    )

    sql = "\n".join(statement for statement, _ in repository.cursor.statements)
    assert "hit_count = hit_count + 1" in sql


class FakeExperienceRepository:
    def __init__(self, *, review_candidates: list[dict] | None = None) -> None:
        self.scope = EntityScope(workspace_id="eval", project_id="entity-resolution", collection_id="tiny", domain="computer-science")
        self.incoming = incoming_entity(self.scope)
        self.existing = stored_entity(self.scope)
        self.review_candidates = review_candidates or []
        self.relation_memory_records: list[dict] = []

    def get_review_task(self, review_task_id: int) -> dict:
        return {
            "id": review_task_id,
            "incoming_entity_id": 42,
            "status": "pending",
            "candidates": self.review_candidates,
        }

    def get_incoming_entity_task(self, incoming_entity_id: int) -> IncomingEntityTask:
        return IncomingEntityTask(id=incoming_entity_id, job_id=5, incoming=self.incoming, status="needs_review", attempt_count=1)

    def get_entity(self, entity_id: int):
        if int(entity_id) == 1:
            return self.existing
        return StoredEntity(
            id=int(entity_id),
            canonical_name="多版本并发控制",
            normalized_name="多版本并发控制",
            entity_type="concept",
            scope=self.scope,
        )

    def upsert_entity(self, incoming: IncomingEntity, matched_entity_id: int | None = None) -> EntityUpsertResult:
        entity_id = int(matched_entity_id or 99)
        return EntityUpsertResult(entity_id=entity_id, canonical_name=incoming.name, created=matched_entity_id is None, attribute_count=0, evidence_count=0)

    def upsert_aliases_for_entity(self, *args, **kwargs) -> None:
        return None

    def record_resolution_audit(self, *args, **kwargs) -> None:
        return None

    def finish_incoming_entity(self, *args, **kwargs) -> None:
        return None

    def complete_review_task(self, *args, **kwargs) -> None:
        return None

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
        self.relation_memory_records.append(
            {
                "incoming_name": incoming.name,
                "candidate_id": candidate.id,
                "decision": decision,
                "relation_type": relation_type,
                "confidence": confidence,
                "source": source,
                "reason": reason,
            }
        )


class FakeHybridIndex:
    def upsert_entity(self, entity: StoredEntity) -> None:
        return None


class FakeSQLRepository(MySQLEntityRepository):
    def __init__(self, *, existing_entity_ids: set[int] | None = None) -> None:
        self.cursor = FakeCursor()
        self.existing_entity_ids = {1} if existing_entity_ids is None else existing_entity_ids

    def connect(self):
        return FakeConnection(self.cursor)

    def get_entity(self, entity_id: int):
        if int(entity_id) in self.existing_entity_ids:
            return stored_entity()
        return None


class FakeConnection:
    def __init__(self, cursor: "FakeCursor") -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self):
        return self._cursor

    def commit(self) -> None:
        return None


class FakeCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement: str, params: tuple | None = None) -> None:
        self.statements.append((statement, params or ()))


def incoming_entity(scope: EntityScope | None = None) -> IncomingEntity:
    return IncomingEntity(
        name="多版本并发控制",
        normalized_name="多版本并发控制",
        entity_type="concept",
        scope=scope or EntityScope(workspace_id="eval", project_id="entity-resolution", collection_id="tiny", domain="computer-science"),
        identity_description="减少读写阻塞的并发控制机制。",
    )


def stored_entity(scope: EntityScope | None = None) -> StoredEntity:
    return StoredEntity(
        id=1,
        canonical_name="MVCC",
        normalized_name="mvcc",
        entity_type="concept",
        scope=scope or EntityScope(workspace_id="eval", project_id="entity-resolution", collection_id="tiny", domain="computer-science"),
        aliases=["多版本并发控制"],
        identity_description="数据库多版本并发控制机制。",
    )

