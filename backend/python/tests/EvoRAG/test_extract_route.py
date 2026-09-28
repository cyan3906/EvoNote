import asyncio

from app.EvoRAG.entity_store.models import EntityIngestQueueResult
from app.EvoRAG.models import BlockEntityExtraction, ExtractedEntity, EvoRAGPreprocessResult, TextBlock
from app.api.routes import evorag


def run(coro):
    return asyncio.run(coro)


def test_extract_route_preprocesses_text_without_ingest(monkeypatch) -> None:
    calls: list[str] = []

    class FakeProcessor:
        async def preprocess(self, text: str) -> EvoRAGPreprocessResult:
            calls.append(text)
            block = TextBlock(block_index=0, heading="Deadlock", l1_text=text)
            return EvoRAGPreprocessResult(
                input_text=text,
                blocks=[
                    BlockEntityExtraction(
                        block=block,
                        entities=[ExtractedEntity(name="死锁", entity_type="concept")],
                    )
                ],
            )

    monkeypatch.setattr(evorag, "EvoRAGProcessor", FakeProcessor)

    result = run(evorag.extract_text(evorag.EvoRAGIngestRequest(text="死锁的必要条件包括互斥。")))

    assert calls == ["死锁的必要条件包括互斥。"]
    assert result.entity_count == 1
    assert result.blocks[0].entities[0].name == "死锁"


def test_ingest_route_queues_preprocess_result_without_sync_ingest(monkeypatch) -> None:
    processor_calls: list[str] = []
    queue_calls: list[EvoRAGPreprocessResult] = []

    class FakeProcessor:
        async def preprocess(self, text: str) -> EvoRAGPreprocessResult:
            processor_calls.append(text)
            block = TextBlock(block_index=0, heading="HTTP/1.1", l1_text=text)
            return EvoRAGPreprocessResult(
                input_text=text,
                blocks=[
                    BlockEntityExtraction(
                        block=block,
                        entities=[ExtractedEntity(name="HTTP/1.1", entity_type="protocol")],
                    )
                ],
            )

    class FakeIngestor:
        def __init__(self, scope) -> None:
            self.scope = scope

        async def queue(
            self,
            preprocess: EvoRAGPreprocessResult,
            *,
            source_note_id: str = "",
            task_name: str = "",
        ) -> EntityIngestQueueResult:
            queue_calls.append(preprocess)
            return EntityIngestQueueResult(
                job_id=42,
                status="queued",
                queued_count=1,
                incoming_entity_ids=[1001],
            )

        async def ingest(self, preprocess: EvoRAGPreprocessResult):
            raise AssertionError("sync ingest should not run in stage 2")

    monkeypatch.setattr(evorag, "EvoRAGProcessor", FakeProcessor)
    monkeypatch.setattr(evorag, "EntityIngestor", FakeIngestor)

    result = run(evorag.ingest_text(evorag.EvoRAGIngestRequest(text="HTTP/1.1 是 HTTP 协议版本。")))

    assert processor_calls == ["HTTP/1.1 是 HTTP 协议版本。"]
    assert len(queue_calls) == 1
    assert result.job_id == 42
    assert result.status == "queued"
    assert result.queued_count == 1
    assert result.incoming_entity_ids == [1001]
    assert result.ingest_results == []
