import asyncio

from app.EvoRAG.models import TextBlock
from app.EvoRAG.prompts import ENTITY_EXTRACTION_SYSTEM_PROMPT
from app.EvoRAG.services.entity_extractor import ENTITY_ADMISSION_THRESHOLD, EntityExtractor


class FakeLLM:
    def __init__(self, responses: dict[int, dict | Exception]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def chat_json(self, *, system_prompt: str, user_payload: dict, operation_name: str) -> dict:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_payload": user_payload,
                "operation_name": operation_name,
            }
        )
        block_index = user_payload["block"]["block_index"]
        response = self.responses[block_index]
        if isinstance(response, Exception):
            raise response
        return response


def run(coro):
    return asyncio.run(coro)


def block(index: int, text: str = "MVCC uses undo log.") -> TextBlock:
    return TextBlock(block_index=index, heading=f"Block {index + 1}", l1_text=text)


def admission_score(
    definable: float = 0.9,
    query_entry: float = 0.9,
    independent_scope: float = 0.9,
    stable_relations: float = 0.9,
    key_sentence: float = 0.9,
) -> dict:
    aggregate = (definable + query_entry + independent_scope + stable_relations + key_sentence) / 5
    return {
        "definable": definable,
        "query_entry": query_entry,
        "independent_scope": independent_scope,
        "stable_relations": stable_relations,
        "key_sentence": key_sentence,
        "aggregate_score": aggregate,
        "rationale": "candidate is an independent knowledge entry",
    }


def extraction_response(name: str = "MVCC") -> dict:
    return {
        "entities": [
            {
                "name": name,
                "entity_type": "concept",
                "aliases": ["mvcc", "MVCC", "mvcc"],
                "identity_description": "Multi-version concurrency control.",
                "admission_score": admission_score(),
                "attributes": {
                    "definition": [
                        {
                            "value": "MVCC is a concurrency control mechanism.",
                            "evidence": "MVCC is a concurrency control mechanism.",
                            "confidence": 0.91,
                        }
                    ],
                    "purpose": [
                        {
                            "value": "Reduce read-write blocking.",
                            "source_text": "read-write blocking",
                            "confidence": 0.82,
                        }
                    ],
                },
            }
        ],
        "warnings": [],
    }


def test_entity_extractor_parses_entities_attributes_and_legacy_evidence() -> None:
    llm = FakeLLM({0: extraction_response()})
    text_block = block(0)

    result = run(EntityExtractor(llm).extract_one(text_block))

    assert result.block == text_block
    assert llm.calls[0]["system_prompt"] == ENTITY_EXTRACTION_SYSTEM_PROMPT
    assert llm.calls[0]["user_payload"]["block"] == text_block.model_dump()
    assert "definition" in llm.calls[0]["user_payload"]["attribute_schema"]
    assert llm.calls[0]["user_payload"]["entity_admission_threshold"] == ENTITY_ADMISSION_THRESHOLD
    assert llm.calls[0]["user_payload"]["entity_admission_dimensions"] == [
        "definable",
        "query_entry",
        "independent_scope",
        "stable_relations",
        "key_sentence",
    ]

    entity = result.entities[0]
    assert entity.name == "MVCC"
    assert entity.entity_type == "concept"
    assert entity.aliases == ["mvcc", "MVCC"]
    assert entity.identity_description == "Multi-version concurrency control."
    assert entity.admission_score.aggregate_score == 0.9

    definition = entity.attributes.definition[0]
    assert definition.value == "MVCC is a concurrency control mechanism."
    assert definition.evidence == "MVCC is a concurrency control mechanism."
    assert definition.confidence == 0.91

    purpose = entity.attributes.purpose[0]
    assert purpose.value == "Reduce read-write blocking."
    assert purpose.evidence == "read-write blocking"
    assert purpose.confidence == 0.82


def test_entity_extractor_filters_low_admission_score_candidates() -> None:
    llm = FakeLLM(
        {
            0: {
                "entities": [
                    {
                        "name": "CI（持续集成）",
                        "entity_type": "process",
                        "aliases": ["CI"],
                        "identity_description": "Continuous integration process.",
                        "admission_score": admission_score(),
                        "attributes": {
                            "definition": [
                                {
                                    "value": "CI means frequently merging code into the main branch.",
                                    "evidence": "核心思想：频繁把多人写的代码合并到主干仓库",
                                    "confidence": 0.9,
                                }
                            ],
                            "mechanism": [
                                {
                                    "value": "Pull code, install dependencies, build, test, and report.",
                                    "evidence": "拉取最新代码 安装依赖 编译 / 打包 自动执行测试 生成报告",
                                    "confidence": 0.85,
                                }
                            ],
                        },
                    },
                    {
                        "name": "拉取最新代码",
                        "entity_type": "method",
                        "aliases": [],
                        "identity_description": "Pipeline step.",
                        "admission_score": admission_score(
                            definable=0.1,
                            query_entry=0.2,
                            independent_scope=0.1,
                            stable_relations=0.1,
                            key_sentence=0.2,
                        ),
                        "attributes": {},
                    },
                ],
                "warnings": [],
            }
        }
    )

    result = run(EntityExtractor(llm).extract_one(block(0, "CI pipeline steps.")))

    assert [entity.name for entity in result.entities] == ["CI（持续集成）"]
    assert result.warnings == ["dropped non-entity candidate: 拉取最新代码 (admission_score=0.140 < 0.70)"]


def test_extract_many_isolates_single_block_failure() -> None:
    llm = FakeLLM(
        {
            0: extraction_response("HTTP/2"),
            1: RuntimeError("model timeout"),
            2: extraction_response("Frame"),
        }
    )
    blocks = [block(0, "HTTP/2 multiplexing."), block(1, "Broken block."), block(2, "Frame data.")]

    results = run(EntityExtractor(llm).extract_many(blocks))

    assert [result.block.block_index for result in results] == [0, 1, 2]
    assert [entity.name for entity in results[0].entities] == ["HTTP/2"]
    assert results[1].entities == []
    assert results[1].warnings == ["entity extraction failed: model timeout"]
    assert [entity.name for entity in results[2].entities] == ["Frame"]
