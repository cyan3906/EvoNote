import asyncio

from pydantic import ValidationError

from app.EvoRAG.constants import ATTRIBUTE_TYPES
from app.EvoRAG.llm import EvoRAGLLMClient
from app.EvoRAG.models import BlockEntityExtraction, BlockExtractionResult, TextBlock
from app.EvoRAG.prompts import ENTITY_EXTRACTION_SYSTEM_PROMPT


ENTITY_ADMISSION_THRESHOLD = 0.7
ENTITY_ADMISSION_DIMENSIONS = [
    "definable",
    "query_entry",
    "independent_scope",
    "stable_relations",
    "key_sentence",
]


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
                "anchor_entity": block.anchor_entity,
                "candidate_entities": block.candidate_entities,
                "attribute_schema": list(ATTRIBUTE_TYPES),
                "entity_admission_threshold": ENTITY_ADMISSION_THRESHOLD,
                "entity_admission_dimensions": ENTITY_ADMISSION_DIMENSIONS,
            },
            operation_name=f"EvoRAG entity extraction block {block.block_index}",
        )

        try:
            result = BlockExtractionResult.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"block {block.block_index} extraction schema validation failed: {exc}") from exc

        accepted_entities = []
        warnings = list(result.warnings)
        for entity in result.entities:
            score = entity.admission_score.aggregate_score
            if score >= ENTITY_ADMISSION_THRESHOLD:
                accepted_entities.append(entity)
                continue
            warnings.append(
                f"dropped non-entity candidate: {entity.name} "
                f"(admission_score={score:.3f} < {ENTITY_ADMISSION_THRESHOLD:.2f})"
            )

        return BlockEntityExtraction(block=block, entities=accepted_entities, warnings=warnings)
