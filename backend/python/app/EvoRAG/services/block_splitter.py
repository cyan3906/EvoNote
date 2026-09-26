from time import perf_counter

from pydantic import ValidationError

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.llm import EvoRAGLLMClient
from app.EvoRAG.models import BlockSplitResult, TextBlock
from app.EvoRAG.prompts import BLOCK_SPLIT_SYSTEM_PROMPT


class BlockSplitter:
    def __init__(self, llm_client: EvoRAGLLMClient, config: EvoRAGSettings = settings) -> None:
        self.llm = llm_client
        self.config = config

    async def split(self, text: str) -> list[TextBlock]:
        blocks, _ = await self.split_with_timings(text)
        return blocks

    async def split_with_timings(self, text: str) -> tuple[list[TextBlock], dict[str, float]]:
        timings: dict[str, float] = {}
        split_started_at = perf_counter()
        data = await self.llm.chat_json(
            system_prompt=BLOCK_SPLIT_SYSTEM_PROMPT,
            user_payload={
                "text": text,
                "max_blocks": self.config.max_blocks,
                "max_block_chars": self.config.max_block_chars,
            },
            operation_name="EvoRAG block split",
        )
        timings["entity_anchor_split_ms"] = elapsed_ms(split_started_at)

        try:
            result = BlockSplitResult.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"block split result schema validation failed: {exc}") from exc

        normalize_started_at = perf_counter()
        blocks = normalize_blocks(
            result.blocks,
            max_blocks=self.config.max_blocks,
            max_block_chars=self.config.max_block_chars,
        )
        timings["physical_chunk_split_ms"] = elapsed_ms(normalize_started_at)
        if not blocks:
            raise ValueError("block split returned no usable blocks")
        return blocks, timings


def normalize_blocks(blocks: list[TextBlock], *, max_blocks: int, max_block_chars: int) -> list[TextBlock]:
    normalized: list[TextBlock] = []
    for index, block in enumerate(blocks[: max(1, max_blocks)]):
        l1_text = block.l1_text.strip()
        if not l1_text:
            continue
        chunks = split_text_by_max_chars(l1_text, max_block_chars=max_block_chars)
        parent_output_index = len(normalized)
        chunk_count = len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            if not chunk:
                continue
            heading = block.heading or block.anchor_entity or f"Block {index + 1}"
            if chunk_count > 1:
                heading = f"{heading} - chunk {chunk_index + 1}"
            char_start, char_end = chunk_char_range(block, l1_text, chunk, chunk_index, max_block_chars)
            normalized.append(
                TextBlock(
                    block_index=len(normalized),
                    heading=heading,
                    anchor_entity=block.anchor_entity,
                    candidate_entities=block.candidate_entities,
                    l1_text=chunk,
                    split_reason="max_chars_chunk" if chunk_count > 1 else block.split_reason or "entity_anchor",
                    anchor_confidence=block.anchor_confidence,
                    parent_block_index=parent_output_index if chunk_count > 1 else block.parent_block_index,
                    chunk_index=chunk_index,
                    chunk_count=chunk_count,
                    char_start=char_start,
                    char_end=char_end,
                )
            )
    return normalized


def split_text_by_max_chars(text: str, *, max_block_chars: int) -> list[str]:
    clean = text.strip()
    if max_block_chars <= 0 or len(clean) <= max_block_chars:
        return [clean]

    chunks: list[str] = []
    cursor = 0
    while cursor < len(clean):
        end = min(cursor + max_block_chars, len(clean))
        if end < len(clean):
            natural_end = find_natural_split(clean, cursor, end)
            if natural_end > cursor:
                end = natural_end
        chunk = clean[cursor:end].strip()
        if chunk:
            chunks.append(chunk)
        cursor = end
        while cursor < len(clean) and clean[cursor].isspace():
            cursor += 1
    return chunks or [clean[:max_block_chars].strip()]


def find_natural_split(text: str, start: int, hard_end: int) -> int:
    min_end = start + max(1, int((hard_end - start) * 0.6))
    for separator in ("\n\n", "\n", "。", "；", ";", "，", ",", " "):
        position = text.rfind(separator, start, hard_end)
        if position >= min_end:
            return position + len(separator)
    return hard_end


def chunk_char_range(block: TextBlock, full_text: str, chunk: str, chunk_index: int, max_block_chars: int) -> tuple[int, int]:
    if block.char_start < 0:
        return -1, -1

    if max_block_chars <= 0:
        return block.char_start, block.char_start + len(chunk)

    approximate_start = min(chunk_index * max_block_chars, len(full_text))
    found_at = full_text.find(chunk, max(0, approximate_start - max_block_chars))
    if found_at < 0:
        found_at = approximate_start
    char_start = block.char_start + found_at
    return char_start, char_start + len(chunk)


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
