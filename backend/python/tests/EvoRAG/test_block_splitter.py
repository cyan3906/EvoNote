import asyncio
from types import SimpleNamespace

import pytest
from conftest import *

from app.EvoRAG.models import TextBlock
from app.EvoRAG.prompts import BLOCK_SPLIT_SYSTEM_PROMPT
from app.EvoRAG.services.block_splitter import BlockSplitter, normalize_blocks


class FakeLLM:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def chat_json(self, *, system_prompt: str, user_payload: dict, operation_name: str) -> dict:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "operation_name": operation_name,
            }
        )
        return self.response


def run(coro):
    return asyncio.run(coro)


def splitter_config(*, max_blocks: int = 4, max_block_chars: int = 20) -> SimpleNamespace:
    return SimpleNamespace(max_blocks=max_blocks, max_block_chars=max_block_chars)


def test_block_splitter_passes_text_and_limits_to_llm() -> None:
    llm = FakeLLM(
        {
            "blocks": [
                {
                    "block_index": 0,
                    "heading": "Overview",
                    "l1_text": "HTTP/2 multiplexing overview.",
                }
            ]
        }
    )
    config = splitter_config(max_blocks=3, max_block_chars=80)

    blocks = run(BlockSplitter(llm, config).split("input text"))

    assert len(blocks) == 1
    assert llm.calls == [
        {
            "system_prompt": BLOCK_SPLIT_SYSTEM_PROMPT,
            "user_payload": {
                "text": "input text",
                "max_blocks": 3,
                "max_block_chars": 80,
            },
            "operation_name": "EvoRAG block split",
        }
    ]


def test_normalize_blocks_drops_empty_blocks_chunks_long_blocks_and_reindexes() -> None:
    blocks = [
        TextBlock(
            block_index=99,
            heading="CI",
            anchor_entity="CI（持续集成）",
            candidate_entities=["CI（持续集成）", "流水线"],
            l1_text="pull code。install deps。run tests",
            split_reason="definition_sentence",
            anchor_confidence=0.92,
            char_start=10,
        ),
        TextBlock(block_index=100, heading="   ", l1_text="   "),
        TextBlock(block_index=101, heading="Given heading", l1_text="Second block"),
    ]

    normalized = normalize_blocks(blocks, max_blocks=5, max_block_chars=14)

    assert [block.block_index for block in normalized] == [0, 1, 2, 3]
    assert [block.heading for block in normalized] == [
        "CI - chunk 1",
        "CI - chunk 2",
        "CI - chunk 3",
        "Given heading",
    ]
    assert [block.l1_text for block in normalized] == [
        "pull code。",
        "install deps。",
        "run tests",
        "Second block",
    ]
    assert [block.anchor_entity for block in normalized[:3]] == ["CI（持续集成）"] * 3
    assert [block.candidate_entities for block in normalized[:3]] == [["CI（持续集成）", "流水线"]] * 3
    assert [block.split_reason for block in normalized[:3]] == ["max_chars_chunk"] * 3
    assert [block.parent_block_index for block in normalized[:3]] == [0, 0, 0]
    assert [block.chunk_index for block in normalized[:3]] == [0, 1, 2]
    assert [block.chunk_count for block in normalized[:3]] == [3, 3, 3]


def test_block_splitter_raises_when_llm_schema_is_invalid() -> None:
    llm = FakeLLM({"blocks": [{"block_index": 0, "heading": "Missing text"}]})

    with pytest.raises(ValueError, match="schema validation failed"):
        run(BlockSplitter(llm, splitter_config()).split("input text"))


def test_block_splitter_raises_when_no_usable_blocks_are_returned() -> None:
    llm = FakeLLM({"blocks": [{"block_index": 0, "heading": "", "l1_text": "   "}]})

    with pytest.raises(ValueError, match="no usable blocks"):
        run(BlockSplitter(llm, splitter_config()).split("input text"))


if __name__ == "__main__":
    pytest.main([__file__])
