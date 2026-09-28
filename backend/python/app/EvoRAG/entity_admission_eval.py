from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.llm import EvoRAGLLMClient
from app.EvoRAG.models import EvoRAGPreprocessResult
from app.EvoRAG.services import EvoRAGProcessor


BACKEND_PYTHON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_DIR = BACKEND_PYTHON_ROOT / "tests" / "EvoRAG" / "fixtures" / "entity_admission"
DEFAULT_OUTPUT_PATH = BACKEND_PYTHON_ROOT / "tests" / "EvoRAG" / "results" / "entity_admission_eval_report.json"

EVAL_JUDGE_SYSTEM_PROMPT = """
你是 EvoRAG 实体准入评测裁判。你会收到一条 gold 数据和当前 EvoRAG 抽取预测。

评测目标：
- 以 gold 为标准，判断预测实体、非实体提升、预测属性是否合理。
- 不要求文字完全一致；语义等价、同义表达、缩写/全称可以匹配。
- 漏掉 gold 属性影响较小，但预测属性如果在 gold 中完全找不到对应事实，需要标为 unmatched。
- 如果预测属性和 gold 或原文明确相反，标为 contradictory。
- 如果预测属性语义匹配 gold，但属性 bucket 不同，status 仍为 matched，bucket_correct=false。
- 如果 gold 的 expected_non_entities 被预测为实体或等价实体，promoted=true。

只输出 JSON，不要 Markdown，不要解释。

JSON 格式：
{
  "case_id": "样本 id",
  "gold_entity_matches": [
    {
      "gold_entity": "gold 实体名",
      "status": "matched | missing",
      "matched_prediction": "匹配到的预测实体，没匹配则空",
      "reason": "一句话理由"
    }
  ],
  "predicted_entity_judgments": [
    {
      "predicted_entity": "预测实体名",
      "status": "correct | incorrect",
      "matched_gold_entity": "匹配到的 gold 实体，错误则空",
      "reason": "一句话理由"
    }
  ],
  "non_entity_judgments": [
    {
      "text": "gold 中不应成为实体的候选",
      "promoted": false,
      "predicted_entity": "如果被提升，写匹配的预测实体",
      "reason": "一句话理由"
    }
  ],
  "predicted_attribute_judgments": [
    {
      "predicted_entity": "预测属性所属实体",
      "predicted_attribute": "预测属性文本",
      "bucket": "definition | purpose | core_idea | mechanism | components | constraints | related",
      "status": "matched | unmatched | contradictory",
      "bucket_correct": true,
      "matched_gold_attribute": "匹配到的 gold 属性，未匹配则空",
      "reason": "一句话理由"
    }
  ],
  "gold_attribute_matches": [
    {
      "gold_entity": "gold 实体名",
      "gold_attribute": "gold 属性文本",
      "bucket": "gold 属性 bucket",
      "status": "matched | missing",
      "matched_predicted_attribute": "匹配到的预测属性，没匹配则空",
      "reason": "一句话理由"
    }
  ],
  "errors": [
    {
      "type": "false_promotion | wrong_parent | unmatched_attribute | contradiction | wrong_bucket | missing_entity",
      "severity": "low | medium | high",
      "text": "错误对象",
      "reason": "一句话理由"
    }
  ]
}
""".strip()


def load_fixture_cases(fixture_dir: Path = DEFAULT_FIXTURE_DIR) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(fixture_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.extend(payload.get("cases", []))
    return cases


def preprocess_to_prediction(preprocess: EvoRAGPreprocessResult) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []

    for block_result in preprocess.blocks:
        block = block_result.block.model_dump()
        block_entities = [entity.model_dump() for entity in block_result.entities]
        blocks.append(
            {
                "block": block,
                "entities": block_entities,
                "warnings": list(block_result.warnings),
            }
        )
        for entity in block_entities:
            entities.append(
                {
                    "name": entity.get("name", ""),
                    "entity_type": entity.get("entity_type", ""),
                    "aliases": entity.get("aliases", []),
                    "identity_description": entity.get("identity_description", ""),
                    "attributes": entity.get("attributes", {}),
                    "block_index": block.get("block_index", -1),
                    "block_heading": block.get("heading", ""),
                    "anchor_entity": block.get("anchor_entity", ""),
                }
            )

    return {
        "input_text": preprocess.input_text,
        "blocks": blocks,
        "entities": entities,
        "timings": preprocess.timings,
    }


async def judge_case(
    case: dict[str, Any],
    prediction: dict[str, Any],
    *,
    llm_client: EvoRAGLLMClient,
    config: EvoRAGSettings = settings,
) -> dict[str, Any]:
    data = await llm_client.chat_json(
        system_prompt=EVAL_JUDGE_SYSTEM_PROMPT,
        user_payload={
            "case_id": case["case_id"],
            "category": case["category"],
            "input_text": case["input_text"],
            "gold": {
                "expected_entities": case["expected_entities"],
                "expected_non_entities": case["expected_non_entities"],
            },
            "prediction": prediction,
        },
        operation_name=f"EvoRAG entity admission eval judge {case['case_id']}",
        model=config.entity_admission_judge_model,
    )
    return normalize_case_judgment(case, data)


def normalize_case_judgment(case: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(data.get("case_id") or case["case_id"]),
        "gold_entity_matches": as_list(data.get("gold_entity_matches")),
        "predicted_entity_judgments": as_list(data.get("predicted_entity_judgments")),
        "non_entity_judgments": as_list(data.get("non_entity_judgments")),
        "predicted_attribute_judgments": as_list(data.get("predicted_attribute_judgments")),
        "gold_attribute_matches": as_list(data.get("gold_attribute_matches")),
        "errors": as_list(data.get("errors")),
    }


async def evaluate_entity_admission_suite(
    *,
    fixture_dir: Path = DEFAULT_FIXTURE_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    config: EvoRAGSettings = settings,
    limit: int | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    cases = load_fixture_cases(fixture_dir)
    if category:
        cases = [case for case in cases if case.get("category") == category]
    if limit is not None:
        cases = cases[: max(0, limit)]

    inference_client = EvoRAGLLMClient(config)
    processor = EvoRAGProcessor(config=config, llm_client=inference_client)
    judge_client = EvoRAGLLMClient(config)
    case_reports: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    inference_usage_records: list[dict[str, Any]] = []
    judge_usage_records: list[dict[str, Any]] = []
    eval_started_at = perf_counter()

    for case in cases:
        case_started_at = perf_counter()
        inference_checkpoint = inference_client.usage_checkpoint()
        judge_checkpoint = judge_client.usage_checkpoint()
        preprocess_started_at = perf_counter()
        preprocess = await processor.preprocess(case["input_text"])
        preprocess_ms = elapsed_ms(preprocess_started_at)
        prediction = preprocess_to_prediction(preprocess)
        judge_started_at = perf_counter()
        judgment = await judge_case(case, prediction, llm_client=judge_client, config=config)
        judge_ms = elapsed_ms(judge_started_at)
        case_inference_usage_records = inference_client.usage_records_since(inference_checkpoint)
        case_judge_usage_records = judge_client.usage_records_since(judge_checkpoint)
        inference_usage_records.extend(case_inference_usage_records)
        judge_usage_records.extend(case_judge_usage_records)
        judgments.append({**judgment, "category": case["category"]})
        case_reports.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "prediction": prediction,
                "judgment": judgment,
                "runtime": {
                    "preprocess_ms": preprocess_ms,
                    "judge_ms": judge_ms,
                    "total_case_ms": elapsed_ms(case_started_at),
                },
                "usage": {
                    "inference": summarize_llm_usage(case_inference_usage_records),
                    "judge": summarize_llm_usage(case_judge_usage_records),
                    "total": summarize_llm_usage(case_inference_usage_records + case_judge_usage_records),
                    "llm_calls": {
                        "inference": case_inference_usage_records,
                        "judge": case_judge_usage_records,
                    },
                },
            }
        )

    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for judgment in judgments:
        grouped[str(judgment.get("category", ""))].append(judgment)
    for name, items in grouped.items():
        by_category[name] = compute_metrics(items)

    report = {
        "judge_model": config.entity_admission_judge_model,
        "inference_model": config.inference_model,
        "fixture_dir": str(fixture_dir),
        "total_cases": len(case_reports),
        "runtime": {
            "total_eval_ms": elapsed_ms(eval_started_at),
            "avg_case_ms": safe_rate(
                sum(case["runtime"]["total_case_ms"] for case in case_reports),
                len(case_reports),
            ),
        },
        "cost": {
            "inference": summarize_llm_usage(inference_usage_records),
            "judge": summarize_llm_usage(judge_usage_records),
            "total": summarize_llm_usage(inference_usage_records + judge_usage_records),
        },
        "overall": compute_metrics(judgments),
        "by_category": by_category,
        "cases": case_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def compute_metrics(case_judgments: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "gold_entities": 0,
        "matched_gold_entities": 0,
        "predicted_entities": 0,
        "correct_predicted_entities": 0,
        "gold_non_entities": 0,
        "false_promotions": 0,
        "predicted_attributes": 0,
        "matched_predicted_attributes": 0,
        "unmatched_predicted_attributes": 0,
        "contradictory_predicted_attributes": 0,
        "bucket_checked_predicted_attributes": 0,
        "wrong_bucket_predicted_attributes": 0,
        "gold_attributes": 0,
        "matched_gold_attributes": 0,
    }

    for judgment in case_judgments:
        gold_entities = as_list(judgment.get("gold_entity_matches"))
        counts["gold_entities"] += len(gold_entities)
        counts["matched_gold_entities"] += sum(1 for item in gold_entities if status(item) == "matched")

        predicted_entities = as_list(judgment.get("predicted_entity_judgments"))
        counts["predicted_entities"] += len(predicted_entities)
        counts["correct_predicted_entities"] += sum(1 for item in predicted_entities if status(item) == "correct")

        non_entities = as_list(judgment.get("non_entity_judgments"))
        counts["gold_non_entities"] += len(non_entities)
        counts["false_promotions"] += sum(1 for item in non_entities if bool(item.get("promoted")))

        predicted_attributes = as_list(judgment.get("predicted_attribute_judgments"))
        counts["predicted_attributes"] += len(predicted_attributes)
        for item in predicted_attributes:
            item_status = status(item)
            if item_status == "matched":
                counts["matched_predicted_attributes"] += 1
                counts["bucket_checked_predicted_attributes"] += 1
                if not bool(item.get("bucket_correct")):
                    counts["wrong_bucket_predicted_attributes"] += 1
            elif item_status == "unmatched":
                counts["unmatched_predicted_attributes"] += 1
            elif item_status == "contradictory":
                counts["contradictory_predicted_attributes"] += 1

        gold_attributes = as_list(judgment.get("gold_attribute_matches"))
        counts["gold_attributes"] += len(gold_attributes)
        counts["matched_gold_attributes"] += sum(1 for item in gold_attributes if status(item) == "matched")

    return {
        "entity_precision": safe_rate(counts["correct_predicted_entities"], counts["predicted_entities"]),
        "entity_recall": safe_rate(counts["matched_gold_entities"], counts["gold_entities"]),
        "false_promotion_rate": safe_rate(counts["false_promotions"], counts["gold_non_entities"]),
        "matched_predicted_attribute_rate": safe_rate(
            counts["matched_predicted_attributes"],
            counts["predicted_attributes"],
        ),
        "unmatched_predicted_attribute_rate": safe_rate(
            counts["unmatched_predicted_attributes"],
            counts["predicted_attributes"],
        ),
        "contradictory_predicted_attribute_rate": safe_rate(
            counts["contradictory_predicted_attributes"],
            counts["predicted_attributes"],
        ),
        "wrong_bucket_rate": safe_rate(
            counts["wrong_bucket_predicted_attributes"],
            counts["bucket_checked_predicted_attributes"],
        ),
        "gold_attribute_recall": safe_rate(counts["matched_gold_attributes"], counts["gold_attributes"]),
        "counts": counts,
    }


def as_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def status(item: dict[str, Any]) -> str:
    return str(item.get("status") or "").strip().lower()


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def summarize_llm_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = empty_usage_summary()
    by_model: dict[str, dict[str, Any]] = {}

    for record in records:
        model = str(record.get("model") or "unknown")
        model_summary = by_model.setdefault(model, empty_usage_summary(include_by_model=False))
        add_usage_record(summary, record)
        add_usage_record(model_summary, record)

    finalize_usage_summary(summary)
    for model_summary in by_model.values():
        finalize_usage_summary(model_summary)
    summary["by_model"] = by_model
    return summary


def empty_usage_summary(*, include_by_model: bool = True) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_duration_ms": 0.0,
        "avg_duration_ms": 0.0,
    }
    if include_by_model:
        summary["by_model"] = {}
    return summary


def add_usage_record(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["calls"] += 1
    summary["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
    summary["completion_tokens"] += int(record.get("completion_tokens") or 0)
    summary["total_tokens"] += int(record.get("total_tokens") or 0)
    summary["total_duration_ms"] += float(record.get("duration_ms") or 0.0)


def finalize_usage_summary(summary: dict[str, Any]) -> None:
    calls = int(summary["calls"])
    total_duration_ms = round(float(summary["total_duration_ms"]), 2)
    summary["total_duration_ms"] = total_duration_ms
    summary["avg_duration_ms"] = safe_rate(total_duration_ms, calls)


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoRAG entity-admission live evaluation.")
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(
        evaluate_entity_admission_suite(
            fixture_dir=args.fixtures,
            output_path=args.out,
            limit=args.limit,
            category=args.category,
        )
    )
    print(json.dumps({"output_path": str(args.out), "overall": report["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
