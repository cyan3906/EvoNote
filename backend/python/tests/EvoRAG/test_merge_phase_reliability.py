import asyncio

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.entity_store.ingestor import EntityIngestor
from app.EvoRAG.entity_store.models import EntityIngestQueueResult, EntityScope, IncomingEntity
from app.EvoRAG.entity_store.normalizer import dedupe_extracted_entities
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.models import AttributeBucket, AttributeValue, BlockEntityExtraction, EvoRAGPreprocessResult, ExtractedEntity, TextBlock


def test_queue_only_persists_mysql_job_and_does_not_enqueue_redis() -> None:
    repository = FakeQueueRepository()
    queue_client = ExplodingQueueClient()
    ingestor = EntityIngestor(
        repository=repository,
        config=EvoRAGSettings(_env_file=None),
        queue_client=queue_client,
    )

    result = asyncio.run(ingestor.queue(preprocess_result(), source_note_id="note-123", task_name="merge-test"))

    assert result.incoming_entity_ids == [501]
    assert queue_client.enqueue_many_calls == []
    assert repository.create_job_calls[0]["source_note_id"] == "note-123"


def test_manual_reject_resets_mysql_task_without_reenqueueing_redis() -> None:
    repository = FakeReviewRepository()
    queue_client = ExplodingQueueClient()
    ingestor = EntityIngestor(
        repository=repository,
        config=EvoRAGSettings(_env_file=None),
        queue_client=queue_client,
    )

    result = asyncio.run(ingestor.reject_review_candidates(7, candidate_entity_ids=[3], reason="不是同一实体"))

    assert result["status"] == "requeued"
    assert repository.reset_ids == [42]
    assert queue_client.enqueue_calls == []


def test_dedupe_extracted_entities_preserves_note_and_block_source_on_attributes() -> None:
    incoming = dedupe_extracted_entities(preprocess_result().blocks, scope=scope(), source_note_id="note-123")[0]

    attribute = incoming.attributes[0]

    assert attribute.note_id == "note-123"
    assert attribute.block_id == "note-123:block:8"
    assert attribute.block_index == 8


def test_repository_does_not_overwrite_matched_entity_core_retrieval_fields() -> None:
    repository = MatchedUpdateRepository()
    incoming = IncomingEntity(
        name="MVCC",
        normalized_name="mvcc",
        entity_type="concept",
        scope=scope(),
        aliases=["多版本并发控制"],
        identity_description="很短的新描述",
        description_for_match="name: MVCC\nidentity: 很短的新描述",
        embedding=[0.9, 0.9],
    )

    repository.upsert_entity(incoming, matched_entity_id=7)

    update_sql = repository.cursor.statements[0][0]
    assert "aliases_json" in update_sql
    assert "identity_description" not in update_sql
    assert "description_for_match" not in update_sql
    assert "embedding_json" not in update_sql


class FakeQueueRepository:
    def __init__(self) -> None:
        self.create_job_calls = []

    def create_ingest_job(self, **kwargs) -> EntityIngestQueueResult:
        self.create_job_calls.append(kwargs)
        return EntityIngestQueueResult(job_id=12, status="queued", queued_count=1, incoming_entity_ids=[501])


class FakeReviewRepository:
    def __init__(self) -> None:
        self.reset_ids: list[int] = []
        self.completed = []
        self.rejections = []
        self.relation_memory_records = []

    def get_review_task(self, review_task_id: int) -> dict:
        return {
            "id": review_task_id,
            "incoming_entity_id": 42,
            "status": "pending",
            "candidates": [{"id": 3}],
        }

    def get_incoming_entity_task(self, incoming_entity_id: int):
        from app.EvoRAG.entity_store.models import IncomingEntityTask

        return IncomingEntityTask(id=incoming_entity_id, job_id=99, incoming=incoming_entity(), status="needs_review", attempt_count=1)

    def record_rejections(self, **kwargs) -> None:
        self.rejections.append(kwargs)

    def complete_review_task(self, *args, **kwargs) -> None:
        self.completed.append((args, kwargs))

    def reset_incoming_for_retry(self, incoming_entity_id: int) -> None:
        self.reset_ids.append(incoming_entity_id)

    def get_entity(self, entity_id: int):
        from app.EvoRAG.entity_store.models import StoredEntity

        return StoredEntity(id=int(entity_id), canonical_name="MVCC", normalized_name="mvcc", entity_type="concept", scope=scope())

    def upsert_entity_resolution_relation_memory(self, **kwargs) -> None:
        self.relation_memory_records.append(kwargs)


class ExplodingQueueClient:
    def __init__(self) -> None:
        self.enqueue_calls = []
        self.enqueue_many_calls = []

    def enqueue(self, *args, **kwargs) -> None:
        self.enqueue_calls.append((args, kwargs))
        raise AssertionError("Redis queue should not be used")

    def enqueue_many(self, *args, **kwargs) -> None:
        self.enqueue_many_calls.append((args, kwargs))
        raise AssertionError("Redis queue should not be used")


class MatchedUpdateRepository(MySQLEntityRepository):
    def __init__(self) -> None:
        self.cursor = RecordingCursor()

    def connect(self):
        return FakeConnection(self.cursor)


class FakeConnection:
    def __init__(self, cursor: "RecordingCursor") -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self):
        return self._cursor

    def commit(self) -> None:
        return None


class RecordingCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple]] = []
        self.lastrowid = 0
        self._row = {"id": 21}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement: str, params: tuple | None = None) -> None:
        self.statements.append((statement, params or ()))

    def fetchone(self):
        return self._row


def preprocess_result() -> EvoRAGPreprocessResult:
    return EvoRAGPreprocessResult(
        input_text="MVCC 通过多版本减少读写阻塞。",
        blocks=[
            BlockEntityExtraction(
                block=TextBlock(block_index=8, heading="MVCC", anchor_entity="MVCC", l1_text="MVCC 通过多版本减少读写阻塞。"),
                entities=[
                    ExtractedEntity(
                        name="MVCC",
                        entity_type="concept",
                        identity_description="数据库多版本并发控制机制。",
                        attributes=AttributeBucket(
                            mechanism=[
                                AttributeValue(
                                    value="通过多版本减少读写阻塞",
                                    evidence="MVCC 通过多版本减少读写阻塞。",
                                )
                            ]
                        ),
                    )
                ],
            )
        ],
    )


def incoming_entity() -> IncomingEntity:
    return IncomingEntity(name="MVCC", normalized_name="mvcc", entity_type="concept", scope=scope())


def scope() -> EntityScope:
    return EntityScope(workspace_id="local", project_id="evorag", collection_id="default", domain="general")
