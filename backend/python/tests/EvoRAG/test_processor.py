import asyncio

import pytest

from app.EvoRAG.models import BlockEntityExtraction, ExtractedEntity, TextBlock
from app.EvoRAG.services.processor import EvoRAGProcessor, EvoRAGProcessorError


class FakeBlockSplitter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def split(self, text: str) -> list[TextBlock]:
        self.calls.append(text)
        return [
            TextBlock(block_index=0, heading="First", l1_text="First block"),
            TextBlock(block_index=1, heading="Second", l1_text="Second block"),
        ]


class FakeEntityExtractor:
    def __init__(self) -> None:
        self.calls: list[list[TextBlock]] = []

    async def extract_many(self, blocks: list[TextBlock]) -> list[BlockEntityExtraction]:
        self.calls.append(blocks)
        return [
            BlockEntityExtraction(
                block=blocks[0],
                entities=[ExtractedEntity(name="MVCC", entity_type="concept")],
            ),
            BlockEntityExtraction(
                block=blocks[1],
                entities=[
                    ExtractedEntity(name="Undo log", entity_type="concept"),
                    ExtractedEntity(name="Read view", entity_type="concept"),
                ],
            ),
        ]


def run(coro):
    return asyncio.run(coro)


def make_processor(splitter: FakeBlockSplitter, extractor: FakeEntityExtractor) -> EvoRAGProcessor:
    return EvoRAGProcessor(
        llm_client=object(),
        block_splitter=splitter,
        entity_extractor=extractor,
    )


def test_processor_rejects_empty_input_before_splitting() -> None:
    splitter = FakeBlockSplitter()
    extractor = FakeEntityExtractor()

    with pytest.raises(EvoRAGProcessorError, match="input text is empty"):
        run(make_processor(splitter, extractor).preprocess("   "))

    assert splitter.calls == []
    assert extractor.calls == []


def test_processor_splits_then_extracts_and_returns_preprocess_result() -> None:
    splitter = FakeBlockSplitter()
    extractor = FakeEntityExtractor()

    result = run(make_processor(splitter, extractor).preprocess("  source text  "))

    assert splitter.calls == ["source text"]
    assert len(extractor.calls) == 1
    assert [block.heading for block in extractor.calls[0]] == ["First", "Second"]
    assert result.input_text == "source text"
    assert len(result.blocks) == 2
    assert result.entity_count == 3
    assert result.timings["startup_to_first_block_split_ms"] >= 0
    assert result.timings["entity_anchor_split_ms"] >= 0
    assert result.timings["physical_chunk_split_ms"] == 0
    assert result.timings["entity_extraction_ms"] >= 0
    assert result.timings["total_ms"] >= 0
    assert [entity.name for block in result.blocks for entity in block.entities] == [
        "MVCC",
        "Undo log",
        "Read view",
    ]
