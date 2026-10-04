import asyncio

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.entity_store.models import EntityScope
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.models import BlockEntityExtraction, BlockExtractionFailure, EvoRAGPreprocessResult, ExtractedEntity, TextBlock
from app.EvoRAG.prompts import BLOCK_SPLIT_SYSTEM_PROMPT
from app.EvoRAG.services.block_splitter import normalize_blocks
from app.EvoRAG.services.entity_extractor import EntityExtractor


def block(index: int, text: str) -> TextBlock:
    return TextBlock(
        block_index=index,
        heading=f"Block {index}",
        anchor_entity=f"Entity {index}",
        candidate_entities=[],
        l1_text=text,
        split_reason="heading",
        anchor_confidence=0.9,
    )


class ControlledExtractor(EntityExtractor):
    def __init__(self, *, fail_indexes: set[int] | None = None, delay: float = 0.0, max_concurrency: int = 4) -> None:
        self.config = EvoRAGSettings(_env_file=None, max_concurrency=max_concurrency)
        self.fail_indexes = fail_indexes or set()
        self.delay = delay
        self.running = 0
        self.max_seen = 0

    async def extract_one(self, item: TextBlock) -> BlockEntityExtraction:
        self.running += 1
        self.max_seen = max(self.max_seen, self.running)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if item.block_index in self.fail_indexes:
                raise RuntimeError(f"extract failed for block {item.block_index}")
            return BlockEntityExtraction(
                block=item,
                entities=[ExtractedEntity(name=item.anchor_entity, entity_type="concept")],
                warnings=[],
            )
        finally:
            self.running -= 1


def test_extract_many_with_failures_keeps_failed_blocks_visible_and_records_internal_failures() -> None:
    extractor = ControlledExtractor(fail_indexes={1})

    extractions, failures = asyncio.run(
        extractor.extract_many_with_failures(
            [
                block(0, "Redis 分布式锁通常使用 SET NX EX 获取锁。"),
                block(1, "Redlock 通过多个 Redis 节点投票来提高可靠性。"),
            ]
        )
    )

    assert [item.block.block_index for item in extractions] == [0, 1]
    assert extractions[1].entities == []
    assert "extract failed for block 1" in extractions[1].warnings[0]
    assert [failure.block_index for failure in failures] == [1]
    assert "extract failed for block 1" in failures[0].error
    assert failures[0].l1_text == "Redlock 通过多个 Redis 节点投票来提高可靠性。"


def test_extract_many_respects_configured_concurrency_limit() -> None:
    extractor = ControlledExtractor(delay=0.03, max_concurrency=2)

    asyncio.run(extractor.extract_many([block(index, f"text {index}") for index in range(6)]))

    assert extractor.max_seen <= 2


def test_normalize_blocks_falls_back_to_source_text_when_llm_output_does_not_cover_source_text() -> None:
    source_text = "MVCC 通过保存多个版本来减少读写阻塞。\nUndo Log 保存旧版本数据。"

    blocks = normalize_blocks(
        [block(0, "MVCC 通过保存多个版本来减少读写阻塞。")],
        max_blocks=4,
        max_block_chars=200,
        source_text=source_text,
    )

    assert len(blocks) == 1
    assert blocks[0].l1_text == source_text
    assert blocks[0].split_reason == "fallback"


def test_block_split_prompt_forbids_rewriting_and_requires_full_coverage() -> None:
    assert "禁止修改原文" in BLOCK_SPLIT_SYSTEM_PROMPT
    assert "拼接" in BLOCK_SPLIT_SYSTEM_PROMPT
    assert "覆盖原文" in BLOCK_SPLIT_SYSTEM_PROMPT


def test_preprocess_result_keeps_failures_internal_to_avoid_returning_failed_block_content() -> None:
    result = EvoRAGPreprocessResult(
        input_text="input",
        blocks=[],
        extraction_failures=[
            BlockExtractionFailure(
                block_index=1,
                heading="Redlock",
                anchor_entity="Redlock",
                l1_text="failed block content",
                error="model timeout",
            )
        ],
    )

    assert result.extraction_failures[0].l1_text == "failed block content"
    assert "extraction_failures" not in result.model_dump()
    assert "failed block content" not in result.model_dump_json()


class RecordingCursor:
    def __init__(self) -> None:
        self.statements = []

    def execute(self, query: str, params=None) -> None:
        self.statements.append((query, params))


def test_repository_records_extraction_failures_for_later_reprocessing() -> None:
    cursor = RecordingCursor()
    repository = MySQLEntityRepository(EvoRAGSettings(_env_file=None))

    repository._insert_extraction_failures(
        cursor,
        job_id=12,
        scope=EntityScope(workspace_id="local", project_id="evorag", collection_id="default", domain="general"),
        source_note_id="note-1",
        failures=[
            BlockExtractionFailure(
                block_index=2,
                heading="Redlock",
                anchor_entity="Redlock",
                l1_text="Redlock 通过多个 Redis 节点投票。",
                error="entity extraction failed: timeout",
            )
        ],
    )

    query, params = cursor.statements[0]
    assert "evorag_entity_extraction_failures" in query
    assert params == (
        12,
        "local",
        "evorag",
        "default",
        "general",
        "note-1",
        2,
        "Redlock",
        "Redlock",
        "Redlock 通过多个 Redis 节点投票。",
        "entity extraction failed: timeout",
    )
