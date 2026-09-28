import asyncio
from types import SimpleNamespace

from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.models import CandidateEntity, EntityScope, IncomingEntity, IncomingEntityTask, StoredEntity
from app.EvoRAG.entity_store.resolver import EntityResolver


def run(coro):
    return asyncio.run(coro)


def config() -> SimpleNamespace:
    return SimpleNamespace(
        api_base_url="",
        api_key="",
        timeout_seconds=1,
        default_workspace_id="local",
        default_project_id="evorag",
        default_collection_id="default",
        default_domain="general",
        entity_resolution_top_k=3,
        entity_resolution_auto_match_threshold=0.9,
        entity_resolution_llm_threshold=0.7,
        entity_resolution_manual_threshold=0.6,
        entity_merge_max_concurrency=1,
    )


def incoming_entity(name: str = "HTTP/1") -> IncomingEntity:
    return IncomingEntity(
        name=name,
        normalized_name=name.lower(),
        entity_type="protocol",
        scope=EntityScope(),
        identity_description=f"{name} protocol",
        description_for_match=f"{name} protocol",
    )


def stored_entity(entity_id: int, name: str) -> StoredEntity:
    return StoredEntity(
        id=entity_id,
        canonical_name=name,
        normalized_name=name.lower(),
        entity_type="protocol",
        scope=EntityScope(),
        identity_description=f"{name} protocol",
        description_for_match=f"{name} protocol",
    )


class FakeHybridIndex:
    def __init__(self, candidates: list[CandidateEntity]) -> None:
        self.candidates = candidates

    def search(self, incoming: IncomingEntity, *, top_k: int) -> list[CandidateEntity]:
        return self.candidates[:top_k]


def test_resolver_filters_manually_rejected_candidates() -> None:
    rejected = stored_entity(1, "HTTP/1.0")
    accepted = stored_entity(2, "HTTP/1.1")
    incoming = incoming_entity()
    resolver = EntityResolver(
        existing_entities=[rejected, accepted],
        config=config(),
        hybrid_index=FakeHybridIndex(
            [
                CandidateEntity(entity=rejected, score=0.88, rank=1, source="fused"),
                CandidateEntity(entity=accepted, score=0.77, rank=2, source="fused"),
            ]
        ),
        alias_lookup=lambda item: None,
        rejection_lookup=lambda item: {1},
    )

    decision = run(resolver.resolve_one(incoming))

    assert decision.decision == "ambiguous"
    assert [candidate.entity.id for candidate in decision.candidates] == [2]
    assert decision.score == 0.77


class FakeEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeRepository:
    def __init__(self, existing: StoredEntity) -> None:
        self.existing = existing
        self.audit_calls: list[tuple[str, int | None]] = []
        self.review_calls: list[tuple[int, str, int]] = []

    def list_entities(self, scope: EntityScope) -> list[StoredEntity]:
        return [self.existing]

    def find_alias_for_incoming(self, incoming: IncomingEntity) -> None:
        return None

    def rejected_candidate_ids(self, incoming: IncomingEntity) -> set[int]:
        return set()

    def record_resolution_audit(self, incoming: IncomingEntity, decision: str, matched_entity_id: int | None, score: float, reason: str) -> None:
        self.audit_calls.append((decision, matched_entity_id))

    def create_review_task(self, *, incoming_entity_id: int, task: IncomingEntityTask, candidates: list[CandidateEntity], reason: str) -> int:
        self.review_calls.append((incoming_entity_id, reason, len(candidates)))
        return 99


def test_ingestor_creates_review_task_for_ambiguous_decision() -> None:
    existing = stored_entity(10, "HTTP/1.1")
    repository = FakeRepository(existing)
    ingestor = EntityIngestor(repository=repository, config=config())
    ingestor.embedding_client = FakeEmbeddingClient()
    ingestor.hybrid_index = FakeHybridIndex([CandidateEntity(entity=existing, score=0.7, rank=1, source="fused")])
    task = IncomingEntityTask(
        id=31,
        job_id=8,
        incoming=incoming_entity(),
        status="processing",
        attempt_count=1,
    )

    status, decision = run(ingestor.process_incoming_task(task))

    assert status == "needs_review"
    assert decision.decision == "ambiguous"
    assert repository.audit_calls == [("ambiguous", None)]
    assert repository.review_calls == [(31, "candidate score below auto-match threshold; manual review required", 1)]


class FakeReviewRepository:
    def __init__(self) -> None:
        self.incoming = incoming_entity()
        self.recorded_rejections: list[int] = []
        self.completed: list[tuple[int, str, str]] = []
        self.reset_ids: list[int] = []

    def get_review_task(self, review_task_id: int) -> dict[str, object]:
        return {"id": review_task_id, "incoming_entity_id": 41, "status": "pending"}

    def get_incoming_entity_task(self, incoming_entity_id: int) -> IncomingEntityTask:
        return IncomingEntityTask(
            id=incoming_entity_id,
            job_id=9,
            incoming=self.incoming,
            status="needs_review",
            attempt_count=1,
        )

    def record_rejections(self, *, incoming: IncomingEntity, candidate_entity_ids: list[int], reason: str, decided_by: str = "manual") -> None:
        self.recorded_rejections.extend(candidate_entity_ids)

    def complete_review_task(
        self,
        review_task_id: int,
        *,
        status: str,
        decision: str,
        decided_entity_id: int | None = None,
        decided_by: str = "manual",
        reason: str = "",
    ) -> None:
        self.completed.append((review_task_id, status, decision))

    def reset_incoming_for_retry(self, incoming_entity_id: int) -> None:
        self.reset_ids.append(incoming_entity_id)


class FakeQueue:
    def __init__(self) -> None:
        self.enqueued: list[tuple[int, int]] = []

    def enqueue(self, incoming_entity_id: int, *, job_id: int = 0) -> str:
        self.enqueued.append((incoming_entity_id, job_id))
        return "1-0"


def test_reject_review_candidates_records_negative_sample_and_requeues() -> None:
    repository = FakeReviewRepository()
    ingestor = EntityIngestor(repository=repository, config=config())
    ingestor.queue_client = FakeQueue()

    result = run(
        ingestor.reject_review_candidates(
            12,
            candidate_entity_ids=[3, 3, 5],
            reason="not same entity",
            decided_by="tester",
        )
    )

    assert repository.recorded_rejections == [3, 5]
    assert repository.completed == [(12, "rejected", "rejected")]
    assert repository.reset_ids == [41]
    assert ingestor.queue_client.enqueued == [(41, 9)]
    assert result["status"] == "requeued"
