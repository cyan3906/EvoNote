import asyncio

from pydantic import ValidationError

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.constants import ATTRIBUTE_TYPES
from app.EvoRAG.llm import EvoRAGLLMClient
from app.EvoRAG.models import BlockEntityExtraction, BlockExtractionFailure, BlockExtractionResult, EntityAdmissionJudgeScore, ExtractedEntity, TextBlock
from app.EvoRAG.prompts import ENTITY_EXTRACTION_SYSTEM_PROMPT


ENTITY_ADMISSION_THRESHOLD = 0.7
ENTITY_ADMISSION_DIMENSIONS = [
    "definable",
    "query_entry",
    "independent_scope",
    "stable_relations",
    "key_sentence",
]

ENTITY_ADMISSION_JUDGE_SYSTEM_PROMPT = """
你是 EvoRAG 的独立实体准入裁判。抽取模型已经给出一个候选实体，你只判断它是否应该成为长期实体记忆节点。

实体定义：
- 实体是可以长期累积知识、被多篇文本新增/补充/修正/冲突的稳定知识对象。
- 标题、名词、操作动作、机制小节、条件项或步骤本身不自动等于实体。
- 如果候选更适合挂到 parent/root entity 的 mechanism、constraints、components 或 related 中，不要判为 entity。

正向评分，每项 0.0-1.0：
- identity_boundary：是否有稳定身份边界，能和父概念、步骤、属性区分。
- knowledge_capacity：是否能承载 definition/purpose/mechanism/constraints/related 等多类知识。
- query_entry_value：用户是否自然会以它为查询入口。
- evolution_stability：未来多文本输入时，它是否值得作为长期合并对象。

扣分，每项 0.0-1.0：
- section_role_penalty：是否只是章节标题或局部小节主题。
- action_phrase_penalty：是否主要是动作、步骤、流程阶段或操作短语。
- parent_attribute_penalty：是否更适合作为父实体的属性。
- context_dependency_penalty：脱离上级标题后是否难以独立理解。
- granularity_penalty：粒度是否太细，难以长期积累知识。
- evidence_weak_penalty：原文是否没有真正定义它，只是顺带提到。

决策：
- decision=entity 且 entity_score >= threshold 才应该保留为实体。
- decision 可为 entity、section_topic、attribute、step、condition、mention。
- 不要重新抽取属性，不要改写候选，只输出 JSON。

JSON 格式：
{
  "decision": "entity | section_topic | attribute | step | condition | mention",
  "entity_score": 0.0,
  "positive_scores": {
    "identity_boundary": 0.0,
    "knowledge_capacity": 0.0,
    "query_entry_value": 0.0,
    "evolution_stability": 0.0
  },
  "deductions": {
    "section_role_penalty": 0.0,
    "action_phrase_penalty": 0.0,
    "parent_attribute_penalty": 0.0,
    "context_dependency_penalty": 0.0,
    "granularity_penalty": 0.0,
    "evidence_weak_penalty": 0.0
  },
  "parent_entity": "如果应归属到父实体，填写父实体",
  "attribute_type": "definition | purpose | core_idea | mechanism | components | constraints | related | ",
  "reason": "一句话说明判断依据"
}
""".strip()


class EntityExtractor:
    def __init__(
        self,
        llm_client: EvoRAGLLMClient,
        config: EvoRAGSettings = settings,
        judge_llm_client: EvoRAGLLMClient | None = None,
    ) -> None:
        self.llm = llm_client
        self.config = config
        self.judge_llm = judge_llm_client or llm_client

    async def extract_many(self, blocks: list[TextBlock]) -> list[BlockEntityExtraction]:
        extractions, _failures = await self.extract_many_with_failures(blocks)
        return extractions

    async def extract_many_with_failures(self, blocks: list[TextBlock]) -> tuple[list[BlockEntityExtraction], list[BlockExtractionFailure]]:
        semaphore = asyncio.Semaphore(max(1, int(self.config.max_concurrency)))

        async def run_one(block: TextBlock) -> BlockEntityExtraction:
            async with semaphore:
                return await self.extract_one(block)

        tasks = [run_one(block) for block in blocks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extractions: list[BlockEntityExtraction] = []
        failures: list[BlockExtractionFailure] = []
        for block, result in zip(blocks, results, strict=False):
            if isinstance(result, Exception):
                warning = f"entity extraction failed: {result}"
                extractions.append(
                    BlockEntityExtraction(
                        block=block,
                        entities=[],
                        warnings=[warning],
                    )
                )
                failures.append(
                    BlockExtractionFailure(
                        block_index=block.block_index,
                        heading=block.heading,
                        anchor_entity=block.anchor_entity,
                        l1_text=block.l1_text,
                        error=warning,
                    )
                )
                continue
            extractions.append(result)

        return extractions, failures

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

        accepted_entities: list[ExtractedEntity] = []
        warnings = list(result.warnings)
        for entity in result.entities:
            score = entity.admission_score.aggregate_score
            if score < ENTITY_ADMISSION_THRESHOLD:
                warnings.append(
                    f"dropped non-entity candidate: {entity.name} "
                    f"(admission_score={score:.3f} < {ENTITY_ADMISSION_THRESHOLD:.2f})"
                )
                continue

            if self.config.entity_admission_judge_enabled:
                judge_score = await self.judge_entity(block, entity)
                entity.admission_judge = judge_score
                score = judge_score.entity_score
                if judge_score.decision == "entity" and score >= self.config.entity_admission_judge_threshold:
                    accepted_entities.append(entity)
                    continue
                warnings.append(
                    f"dropped non-entity candidate by judge: {entity.name} "
                    f"(decision={judge_score.decision}, entity_score={score:.3f} < {self.config.entity_admission_judge_threshold:.2f})"
                )
            else:
                accepted_entities.append(entity)

        return BlockEntityExtraction(block=block, entities=accepted_entities, warnings=warnings)

    async def judge_entity(self, block: TextBlock, entity: ExtractedEntity) -> EntityAdmissionJudgeScore:
        data = await self.judge_llm.chat_json(
            system_prompt=ENTITY_ADMISSION_JUDGE_SYSTEM_PROMPT,
            user_payload={
                "block": block.model_dump(),
                "candidate": entity.model_dump(),
                "threshold": self.config.entity_admission_judge_threshold,
            },
            operation_name=f"EvoRAG entity admission judge block {block.block_index} entity {entity.name}",
            model=self.config.entity_admission_judge_model,
        )
        return EntityAdmissionJudgeScore.model_validate(data)
