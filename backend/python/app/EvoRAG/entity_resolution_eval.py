from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import CandidateEntity, EntityScope, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name
from app.EvoRAG.indexes import EntityHybridIndex, EvoRAGEmbeddingClient
from app.EvoRAG.llm import EvoRAGLLMClient
from app.EvoRAG.entity_admission_eval import summarize_llm_usage


BACKEND_PYTHON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_PATH = BACKEND_PYTHON_ROOT / "tests" / "EvoRAG" / "fixtures" / "entity_resolution" / "benchmark_v1.json"
DEFAULT_OUTPUT_PATH = BACKEND_PYTHON_ROOT / "tests" / "EvoRAG" / "results" / "entity_resolution_eval_report.json"
DEFAULT_TOP_K = 5

ENTITY_RESOLUTION_JUDGE_SYSTEM_PROMPT = """
你是 EvoRAG 的实体匹配评估裁判。你会收到一个 incoming entity、混合检索候选，以及 gold expected。

目标：判断 incoming entity 是否应该匹配某个候选实体。

规则：
- decision 只能是 matched、ambiguous 或 new。
- 如果 incoming 与某个候选是同一知识实体，decision=matched，并填写 matched_seed_id。
- 如果 incoming 太短或多义，且多个候选都可能成立，decision=ambiguous，matched_seed_id=null。
- 如果候选里没有同一实体，即使存在相关实体，decision=new，matched_seed_id=null。
- related-but-different 的实体不能判为 matched。
- 不要引入候选列表之外的实体。

只输出 JSON，不要 Markdown，不要解释。

JSON 格式：
{
  "decision": "matched | ambiguous | new",
  "matched_seed_id": "seed-001 或 null",
  "confidence": 0.0,
  "reason": "一句话理由"
}
""".strip()


def load_benchmark(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> dict[str, Any]:
    return json.loads(fixture_path.read_text(encoding="utf-8"))


async def evaluate_entity_resolution_suite(
    *,
    fixture_path: Path = DEFAULT_FIXTURE_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    config: EvoRAGSettings = settings,
    top_k: int = DEFAULT_TOP_K,
    limit: int | None = None,
    category: str | None = None,
    embedding_client: EvoRAGEmbeddingClient | None = None,
    hybrid_index: EntityHybridIndex | None = None,
    judge_client: EvoRAGLLMClient | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    started_at = perf_counter()
    benchmark = load_benchmark(fixture_path)
    runtime_config = config_for_eval_indexes(config, str(benchmark.get("dataset_id") or "entity-resolution"))
    scope = scope_from_payload(benchmark["scope"])
    seed_entities = list(benchmark.get("seed_entities") or [])
    cases = list(benchmark.get("cases") or [])
    if category:
        cases = [case for case in cases if case.get("category") == category]
    if limit is not None:
        cases = cases[: max(0, limit)]

    emit(progress, f"加载数据集 {benchmark.get('dataset_id', '')}: seeds={len(seed_entities)} cases={len(cases)} scope={scope.collection_id}")

    embedding_client = embedding_client or EvoRAGEmbeddingClient(runtime_config)
    hybrid_index = hybrid_index or EntityHybridIndex(runtime_config)
    judge_client = judge_client or EvoRAGLLMClient(runtime_config)

    id_maps = build_seed_id_maps(seed_entities)
    stored_entities = build_stored_seed_entities(seed_entities, scope=scope, id_maps=id_maps)
    seed_texts = [entity_text_for_embedding(entity) for entity in stored_entities]
    seed_vectors = await embedding_client.embed_texts(seed_texts)
    for entity, vector in zip(stored_entities, seed_vectors, strict=False):
        entity.embedding = vector

    for index, entity in enumerate(stored_entities, start=1):
        seed_id = id_maps["seed_id_by_numeric_id"][entity.id]
        emit(progress, f"写入 seed 实体 {index}/{len(stored_entities)}: {seed_id} {entity.canonical_name}")
        hybrid_index.upsert_entity(entity)

    case_reports: list[dict[str, Any]] = []
    judge_usage_records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_started_at = perf_counter()
        emit(progress, f"处理 case {index}/{len(cases)}: {case['case_id']} {case['category']}")
        incoming = build_incoming_entity(case["incoming_entity"], scope=scope)
        incoming.embedding = await embedding_client.embed_text(entity_text_for_embedding(incoming))
        candidates = hybrid_index.search(incoming, top_k=top_k)
        retrieval = build_retrieval_report(candidates, id_maps=id_maps)

        checkpoint = judge_client.usage_checkpoint() if hasattr(judge_client, "usage_checkpoint") else 0
        judgment = await judge_entity_resolution_case(
            case,
            incoming=incoming,
            retrieval=retrieval,
            llm_client=judge_client,
            config=runtime_config,
        )
        case_judge_usage = judge_client.usage_records_since(checkpoint) if hasattr(judge_client, "usage_records_since") else []
        judge_usage_records.extend(case_judge_usage)
        case_reports.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "incoming_entity": case["incoming_entity"],
                "expected": case["expected"],
                "retrieval": retrieval,
                "judgment": judgment,
                "runtime": {"total_case_ms": elapsed_ms(case_started_at)},
                "usage": {"judge": summarize_llm_usage(case_judge_usage)},
            }
        )

    report = {
        "dataset_id": benchmark.get("dataset_id", ""),
        "fixture_path": str(fixture_path),
        "output_path": str(output_path),
        "scope": benchmark["scope"],
        "top_k": top_k,
        "seed_entity_count": len(seed_entities),
        "total_cases": len(case_reports),
        "models": {
            "embedding_model": runtime_config.embedding_model,
            "judge_model": runtime_config.entity_admission_judge_model,
        },
        "indexes": {
            "elasticsearch": runtime_config.es_entity_index,
            "milvus": runtime_config.milvus_entity_collection,
        },
        "runtime": {
            "total_eval_ms": elapsed_ms(started_at),
            "avg_case_ms": safe_rate(sum(item["runtime"]["total_case_ms"] for item in case_reports), len(case_reports)),
        },
        "cost": {"judge": summarize_llm_usage(judge_usage_records)},
        "overall": build_metric_block(case_reports),
        "by_category": build_category_metrics(case_reports),
        "cases": case_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit(progress, f"评估完成: output={output_path}")
    return report


async def judge_entity_resolution_case(
    case: dict[str, Any],
    *,
    incoming: IncomingEntity,
    retrieval: dict[str, Any],
    llm_client: EvoRAGLLMClient,
    config: EvoRAGSettings = settings,
) -> dict[str, Any]:
    data = await llm_client.chat_json(
        system_prompt=ENTITY_RESOLUTION_JUDGE_SYSTEM_PROMPT,
        user_payload={
            "case_id": case["case_id"],
            "category": case["category"],
            "incoming_entity": incoming_snapshot(incoming),
            "retrieved_candidates": retrieval["candidates"],
            "gold_expected": case["expected"],
        },
        operation_name=f"EvoRAG entity resolution eval judge {case['case_id']}",
        model=config.entity_admission_judge_model,
    )
    return normalize_judgment(data)


def normalize_judgment(data: dict[str, Any]) -> dict[str, Any]:
    decision = str(data.get("decision") or "ambiguous").strip().lower()
    if decision not in {"matched", "ambiguous", "new"}:
        decision = "ambiguous"
    matched_seed_id = data.get("matched_seed_id")
    return {
        "decision": decision,
        "matched_seed_id": str(matched_seed_id) if matched_seed_id else None,
        "confidence": float(data.get("confidence") or 0.0),
        "reason": str(data.get("reason") or ""),
    }


def compute_retrieval_metrics(case_reports: list[dict[str, Any]], *, k_values: Sequence[int] = (1, 3, 5)) -> dict[str, Any]:
    counts: dict[str, int] = {
        "matched_cases": 0,
        "ambiguous_cases": 0,
        "ambiguous_covered@3": 0,
        "close_negative_cases": 0,
        "close_negative_top1": 0,
    }
    reciprocal_rank_sum = 0.0
    for k in k_values:
        counts[f"matched_found@{k}"] = 0

    for report in case_reports:
        expected = report.get("expected", {})
        ranked_seed_ids = [str(seed_id) for seed_id in report.get("retrieval", {}).get("ranked_seed_ids", [])]
        decision = str(expected.get("decision") or "")
        matched_seed_id = expected.get("matched_seed_id")
        target_ids = expected_target_ids(expected)
        if target_ids:
            counts["matched_cases"] += 1
            first_rank = first_matching_rank(ranked_seed_ids, target_ids)
            if first_rank:
                reciprocal_rank_sum += 1.0 / first_rank
            for k in k_values:
                if first_rank and first_rank <= k:
                    counts[f"matched_found@{k}"] += 1

        if decision == "ambiguous":
            ambiguous_ids = [str(seed_id) for seed_id in expected.get("ambiguous_seed_ids") or []]
            if ambiguous_ids:
                counts["ambiguous_cases"] += 1
                if set(ambiguous_ids).issubset(set(ranked_seed_ids[:3])):
                    counts["ambiguous_covered@3"] += 1

        close_negative_ids = [str(seed_id) for seed_id in expected.get("must_not_match_seed_ids") or []]
        if close_negative_ids and decision in {"related_but_different", "new", "matched"}:
            if report.get("category") in {"related_but_different", "nil_new_entity"}:
                counts["close_negative_cases"] += 1
                if ranked_seed_ids and ranked_seed_ids[0] in close_negative_ids:
                    counts["close_negative_top1"] += 1

    metrics: dict[str, Any] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = safe_rate(counts[f"matched_found@{k}"], counts["matched_cases"])
    metrics["mrr"] = safe_rate(reciprocal_rank_sum, counts["matched_cases"])
    metrics["ambiguous_coverage@3"] = safe_rate(counts["ambiguous_covered@3"], counts["ambiguous_cases"])
    metrics["close_negative_top1_rate"] = safe_rate(counts["close_negative_top1"], counts["close_negative_cases"])
    metrics["counts"] = counts
    return metrics


def compute_deepseek_decision_metrics(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "total_cases": len(case_reports),
        "correct_decisions": 0,
        "expected_matched_cases": 0,
        "correct_matched_entities": 0,
        "expected_non_matched_cases": 0,
        "false_matches": 0,
        "false_new_cases": 0,
        "clear_cases": 0,
        "false_ambiguous_cases": 0,
        "ambiguous_cases": 0,
        "correct_ambiguous_cases": 0,
    }
    for report in case_reports:
        expected = report.get("expected", {})
        judgment = report.get("judgment", {})
        expected_decision = str(expected.get("decision") or "")
        actual_decision = str(judgment.get("decision") or "")
        expected_seed = expected.get("matched_seed_id")
        actual_seed = judgment.get("matched_seed_id")

        if expected_decision == actual_decision:
            counts["correct_decisions"] += 1
        if expected_decision == "matched":
            counts["expected_matched_cases"] += 1
            if actual_decision == "matched" and actual_seed == expected_seed:
                counts["correct_matched_entities"] += 1
            if actual_decision == "new":
                counts["false_new_cases"] += 1
        else:
            counts["expected_non_matched_cases"] += 1
            if actual_decision == "matched":
                counts["false_matches"] += 1
        if expected_decision in {"matched", "new"}:
            counts["clear_cases"] += 1
            if actual_decision == "ambiguous":
                counts["false_ambiguous_cases"] += 1
        if expected_decision == "ambiguous":
            counts["ambiguous_cases"] += 1
            if actual_decision == "ambiguous":
                counts["correct_ambiguous_cases"] += 1

    return {
        "decision_accuracy": safe_rate(counts["correct_decisions"], counts["total_cases"]),
        "matched_entity_accuracy": safe_rate(counts["correct_matched_entities"], counts["expected_matched_cases"]),
        "false_match_rate": safe_rate(counts["false_matches"], counts["expected_non_matched_cases"]),
        "false_new_rate": safe_rate(counts["false_new_cases"], counts["expected_matched_cases"]),
        "false_ambiguous_rate": safe_rate(counts["false_ambiguous_cases"], counts["clear_cases"]),
        "ambiguous_accuracy": safe_rate(counts["correct_ambiguous_cases"], counts["ambiguous_cases"]),
        "counts": counts,
    }


def build_metric_block(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "retrieval": compute_retrieval_metrics(case_reports),
        "deepseek_judge": compute_deepseek_decision_metrics(case_reports),
    }


def build_category_metrics(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in case_reports:
        grouped[str(report.get("category") or "")].append(report)
    return {category: build_metric_block(items) for category, items in grouped.items()}


def build_seed_id_maps(seed_entities: list[dict[str, Any]]) -> dict[str, dict[Any, Any]]:
    seed_id_by_numeric_id: dict[int, str] = {}
    numeric_id_by_seed_id: dict[str, int] = {}
    for index, seed in enumerate(seed_entities, start=1):
        seed_id = str(seed["seed_id"])
        seed_id_by_numeric_id[index] = seed_id
        numeric_id_by_seed_id[seed_id] = index
    return {
        "seed_id_by_numeric_id": seed_id_by_numeric_id,
        "numeric_id_by_seed_id": numeric_id_by_seed_id,
    }


def build_stored_seed_entities(seed_entities: list[dict[str, Any]], *, scope: EntityScope, id_maps: dict[str, dict[Any, Any]]) -> list[StoredEntity]:
    return [
        StoredEntity(
            id=int(id_maps["numeric_id_by_seed_id"][str(seed["seed_id"])]),
            canonical_name=str(seed["canonical_name"]),
            normalized_name=normalize_name(str(seed["canonical_name"])),
            entity_type=str(seed.get("entity_type") or "concept"),
            scope=scope,
            aliases=[str(alias) for alias in seed.get("aliases") or []],
            identity_description=str(seed.get("identity_description") or ""),
            summary=seed_summary(seed),
            description_for_match=description_for_seed(seed),
        )
        for seed in seed_entities
    ]


def build_incoming_entity(payload: dict[str, Any], *, scope: EntityScope) -> IncomingEntity:
    name = str(payload.get("name") or "")
    return IncomingEntity(
        name=name,
        normalized_name=normalize_name(name),
        entity_type=str(payload.get("entity_type") or ""),
        scope=scope,
        aliases=[str(alias) for alias in payload.get("aliases") or []],
        identity_description=str(payload.get("identity_description") or ""),
        description_for_match=description_for_incoming(payload),
    )


def build_retrieval_report(candidates: list[CandidateEntity], *, id_maps: dict[str, dict[Any, Any]]) -> dict[str, Any]:
    seed_by_id = id_maps["seed_id_by_numeric_id"]
    items: list[dict[str, Any]] = []
    ranked_seed_ids: list[str] = []
    for rank, candidate in enumerate(candidates, start=1):
        seed_id = seed_by_id.get(candidate.entity.id, f"entity-{candidate.entity.id}")
        ranked_seed_ids.append(seed_id)
        items.append(
            {
                "rank": rank,
                "seed_id": seed_id,
                "entity_id": candidate.entity.id,
                "canonical_name": candidate.entity.canonical_name,
                "entity_type": candidate.entity.entity_type,
                "score": candidate.score,
                "source": candidate.source,
                "vector_score": candidate.vector_score,
                "es_score": candidate.es_score,
                "identity_description": candidate.entity.identity_description,
            }
        )
    return {"ranked_seed_ids": ranked_seed_ids, "candidates": items}


def scope_from_payload(payload: dict[str, Any]) -> EntityScope:
    return EntityScope(
        workspace_id=str(payload.get("workspace_id") or "eval"),
        project_id=str(payload.get("project_id") or "entity-resolution"),
        collection_id=str(payload.get("collection_id") or "benchmark-v1"),
        domain=str(payload.get("domain") or "computer-science"),
    )


def config_for_eval_indexes(config: EvoRAGSettings, dataset_id: str) -> EvoRAGSettings:
    suffix = safe_identifier(dataset_id)
    return config.model_copy(
        update={
            "es_entity_index": f"{config.es_entity_index}_eval_{suffix}",
            "milvus_entity_collection": f"{config.milvus_entity_collection}_eval_{suffix}",
        }
    )


def safe_identifier(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in str(value or ""))
    collapsed = "_".join(part for part in normalized.split("_") if part)
    return collapsed or "benchmark"


def seed_summary(seed: dict[str, Any]) -> str:
    attributes = seed.get("attributes") or {}
    definitions = attributes.get("definition") or []
    if definitions:
        return str(definitions[0])
    return str(seed.get("identity_description") or "")[:500]


def description_for_seed(seed: dict[str, Any]) -> str:
    parts = [f"name: {seed.get('canonical_name', '')}", f"type: {seed.get('entity_type', '')}"]
    aliases = [str(alias) for alias in seed.get("aliases") or []]
    if aliases:
        parts.append(f"aliases: {', '.join(aliases)}")
    if seed.get("identity_description"):
        parts.append(f"identity: {seed['identity_description']}")
    attributes = seed.get("attributes") or {}
    for attr_type in ("definition", "purpose", "core_idea", "mechanism", "components", "constraints", "related"):
        values = attributes.get(attr_type) or []
        if values:
            parts.append(f"{attr_type}: {'; '.join(str(value) for value in values)}")
    return "\n".join(parts)


def description_for_incoming(payload: dict[str, Any]) -> str:
    parts = [f"name: {payload.get('name', '')}", f"type: {payload.get('entity_type', '')}"]
    aliases = [str(alias) for alias in payload.get("aliases") or []]
    if aliases:
        parts.append(f"aliases: {', '.join(aliases)}")
    if payload.get("identity_description"):
        parts.append(f"identity: {payload['identity_description']}")
    return "\n".join(parts)


def entity_text_for_embedding(entity: StoredEntity | IncomingEntity) -> str:
    return entity.identity_description or entity.description_for_match or entity.canonical_name if isinstance(entity, StoredEntity) else entity.identity_description or entity.description_for_match or entity.name


def incoming_snapshot(incoming: IncomingEntity) -> dict[str, Any]:
    return {
        "name": incoming.name,
        "normalized_name": incoming.normalized_name,
        "entity_type": incoming.entity_type,
        "aliases": incoming.aliases,
        "identity_description": incoming.identity_description,
        "description_for_match": incoming.description_for_match,
        "scope": incoming.scope.as_dict(),
    }


def expected_target_ids(expected: dict[str, Any]) -> list[str]:
    if str(expected.get("decision") or "") == "matched" and expected.get("matched_seed_id"):
        return [str(expected["matched_seed_id"])]
    return []


def first_matching_rank(ranked_seed_ids: list[str], target_ids: list[str]) -> int | None:
    target_set = set(target_ids)
    for index, seed_id in enumerate(ranked_seed_ids, start=1):
        if seed_id in target_set:
            return index
    return None


def emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress:
        progress(message)


def safe_rate(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator, 6)


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoRAG entity-resolution retrieval + DeepSeek judge evaluation.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def print_progress(message: str) -> None:
        print(message, flush=True)

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=args.fixture,
            output_path=args.out,
            top_k=args.top_k,
            limit=args.limit,
            category=args.category,
            progress=print_progress,
        )
    )
    print(json.dumps({"output_path": str(args.out), "overall": report["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


