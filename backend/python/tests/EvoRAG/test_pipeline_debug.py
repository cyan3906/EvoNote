import asyncio

from app.EvoRAG.entity_store.models import (
    CandidateEntity,
    EntityIngestQueueResult,
    EntityResolutionDecision,
    EntityScope,
    IncomingEntity,
    StoredEntity,
)
from app.EvoRAG.models import AttributeBucket, AttributeValue, BlockEntityExtraction, EvoRAGPreprocessResult, ExtractedEntity, TextBlock
from app.EvoRAG.services.pipeline_debug import run_debug_pipeline


def scope() -> EntityScope:
    return EntityScope(workspace_id="debug", project_id="pipeline-test", collection_id="unit-test", domain="general")


def preprocess_result() -> EvoRAGPreprocessResult:
    block = TextBlock(block_index=0, heading="Pipeline", anchor_entity="Transformer", l1_text="Transformer uses attention.")
    entity = ExtractedEntity(
        name="Transformer",
        entity_type="concept",
        identity_description="A neural network architecture.",
        attributes=AttributeBucket(definition=[AttributeValue(value="Transformer architecture", evidence="Transformer uses attention.")]),
    )
    return EvoRAGPreprocessResult(
        input_text="Transformer uses attention.",
        blocks=[BlockEntityExtraction(block=block, entities=[entity], warnings=["entity warning"])],
        timings={"total_ms": 12.5},
    )


def incoming_entity() -> IncomingEntity:
    return IncomingEntity(
        name="Transformer",
        normalized_name="transformer",
        entity_type="concept",
        scope=scope(),
        aliases=["Transformer 架构"],
        identity_description="A neural network architecture.",
        description_for_match="Transformer architecture",
    )


def stored_entity() -> StoredEntity:
    return StoredEntity(
        id=9,
        canonical_name="Transformer",
        normalized_name="transformer",
        entity_type="concept",
        scope=scope(),
        summary="Existing Transformer entity",
    )


class FakeProcessor:
    async def preprocess(self, text: str) -> EvoRAGPreprocessResult:
        assert text == "Transformer uses attention."
        return preprocess_result()


class FakeIngestor:
    def __init__(self) -> None:
        self.incoming = incoming_entity()
        self.repository = FakeRepository()
        self.processed_ids: list[int] = []

    async def preview_memory_guarded_hybrid(self, result: EvoRAGPreprocessResult, *, source_note_id: str = ""):
        assert result.input_text == "Transformer uses attention."
        assert source_note_id == "note-1"
        decision = EntityResolutionDecision(
            incoming=self.incoming,
            decision="matched",
            matched_entity=stored_entity(),
            score=0.94,
            reason="relation memory direct allow",
            candidates=[CandidateEntity(entity=stored_entity(), score=0.94, rank=1, source="relation_memory")],
            resolution_trace=[
                {
                    "stage": "redis_direct_relation",
                    "status": "allow",
                    "relation": {"candidate_entity": {"id": 9}},
                }
            ],
            resolution_exit_stage="redis_direct_relation",
        )
        return [self.incoming], [decision]

    async def queue(self, result: EvoRAGPreprocessResult, *, source_note_id: str = "", task_name: str = "") -> EntityIngestQueueResult:
        assert result.entity_count == 1
        assert source_note_id == "note-1"
        assert task_name == "pipeline-debug"
        return EntityIngestQueueResult(job_id=42, status="queued", queued_count=1, incoming_entity_ids=[101])

    async def process_incoming_entity_id(self, incoming_entity_id: int, *, worker_id: str) -> str:
        assert worker_id == "debug-pipeline"
        self.processed_ids.append(incoming_entity_id)
        return "auto_merged"


class FakeRepository:
    def get_ingest_job_status(self, job_id: int) -> dict[str, object]:
        assert job_id == 42
        return {"id": 42, "status": "completed", "total_count": 1, "completed_count": 1}


def test_debug_pipeline_returns_extraction_memory_guarded_hybrid_and_merge_stages() -> None:
    ingestor = FakeIngestor()

    payload = asyncio.run(
        run_debug_pipeline(
            "Transformer uses attention.",
            scope=scope(),
            source_note_id="note-1",
            task_name="pipeline-debug",
            processor=FakeProcessor(),
            ingestor=ingestor,
        )
    )

    assert payload["scope"] == scope().as_dict()
    assert list(payload["stages"].keys()) == [
        "stage1_extraction",
        "stage2_memory_guarded_hybrid",
        "stage3_merge_persistence",
    ]
    assert payload["stages"]["stage1_extraction"]["entity_count"] == 1
    assert payload["stages"]["stage1_extraction"]["entity_extraction"]["blocks"][0]["warnings"] == ["entity warning"]
    assert payload["stages"]["stage2_memory_guarded_hybrid"]["incoming_count"] == 1
    assert payload["stages"]["stage2_memory_guarded_hybrid"]["decisions"][0]["decision"] == "matched"
    assert payload["stages"]["stage2_memory_guarded_hybrid"]["decisions"][0]["candidates"][0]["source"] == "relation_memory"
    trace = payload["stages"]["stage2_memory_guarded_hybrid"]["traces"][0]
    assert trace["exit_stage"] == "redis_direct_relation"
    assert trace["final_decision"]["decision"] == "matched"
    assert trace["steps"][0]["stage"] == "redis_direct_relation"
    assert trace["steps"][0]["status"] == "allow"
    assert payload["stages"]["stage3_merge_persistence"]["job_id"] == 42
    assert payload["stages"]["stage3_merge_persistence"]["processed"][0]["status"] == "auto_merged"
    assert ingestor.processed_ids == [101]
