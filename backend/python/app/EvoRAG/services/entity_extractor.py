import asyncio

from pydantic import ValidationError

from app.EvoRAG.constants import ATTRIBUTE_TYPES
from app.EvoRAG.llm import EvoRAGLLMClient
from app.EvoRAG.models import BlockEntityExtraction, BlockExtractionResult, TextBlock
from app.EvoRAG.prompts import ENTITY_EXTRACTION_SYSTEM_PROMPT


class EntityExtractor:
    def __init__(self, llm_client: EvoRAGLLMClient) -> None:
        self.llm = llm_client

    async def extract_many(self, blocks: list[TextBlock]) -> list[BlockEntityExtraction]:
        tasks = [self.extract_one(block) for block in blocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extractions: list[BlockEntityExtraction] = []
        for block, result in zip(blocks, results, strict=False):
            if isinstance(result, Exception):
                extractions.append(
                    BlockEntityExtraction(
                        block=block,
                        entities=[],
                        warnings=[f"entity extraction failed: {result}"],
                    )
                )
                continue
            extractions.append(result)

        return extractions

    async def extract_one(self, block: TextBlock) -> BlockEntityExtraction:
        data = await self.llm.chat_json(
            system_prompt=ENTITY_EXTRACTION_SYSTEM_PROMPT,
            user_payload={
                "block": block.model_dump(),
                "attribute_schema": list(ATTRIBUTE_TYPES),
            },
            operation_name=f"EvoRAG entity extraction block {block.block_index}",
        )

        try:
            result = BlockExtractionResult.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"block {block.block_index} extraction schema validation failed: {exc}") from exc

        return BlockEntityExtraction(block=block, entities=result.entities, warnings=result.warnings)
