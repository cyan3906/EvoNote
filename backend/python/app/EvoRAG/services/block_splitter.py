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
        data = await self.llm.chat_json(
            system_prompt=BLOCK_SPLIT_SYSTEM_PROMPT,
            user_payload={
                "text": text,
                "max_blocks": self.config.max_blocks,
                "max_block_chars": self.config.max_block_chars,
            },
            operation_name="EvoRAG block split",
        )

        try:
            result = BlockSplitResult.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"block split result schema validation failed: {exc}") from exc

        blocks = normalize_blocks(
            result.blocks,
            max_blocks=self.config.max_blocks,
            max_block_chars=self.config.max_block_chars,
        )
        if not blocks:
            raise ValueError("block split returned no usable blocks")
        return blocks


def normalize_blocks(blocks: list[TextBlock], *, max_blocks: int, max_block_chars: int) -> list[TextBlock]:
    normalized: list[TextBlock] = []
    for index, block in enumerate(blocks[: max(1, max_blocks)]):
        l1_text = block.l1_text.strip()
        if not l1_text:
            continue
        if max_block_chars > 0 and len(l1_text) > max_block_chars:
            l1_text = l1_text[:max_block_chars].rstrip()
        normalized.append(
            TextBlock(
                block_index=len(normalized),
                heading=block.heading or f"Block {index + 1}",
                l1_text=l1_text,
            )
        )
    return normalized
