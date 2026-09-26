import asyncio

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
