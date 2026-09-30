from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.entity_store.models import CandidateEntity, EntityRelationMemoryRecord, EntityScope, IncomingEntity, StoredEntity
from app.EvoRAG.entity_store.normalizer import normalize_name
from app.EvoRAG.entity_store.relation_memory import EntityRelationMemory, is_allow_relation, is_reject_relation
from app.EvoRAG.entity_store.repository import MySQLEntityRepository
from app.EvoRAG.indexes import EntityHybridIndex, EvoRAGEmbeddingClient
from app.EvoRAG.llm import EvoRAGLLMClient
from app.EvoRAG.entity_admission_eval import summarize_llm_usage


BACKEND_PYTHON_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_PATH = BACKEND_PYTHON_ROOT / "tests" / "EvoRAG" / "fixtures" / "entity_resolution" / "benchmark_v1.json"
DEFAULT_OUTPUT_PATH = BACKEND_PYTHON_ROOT / "tests" / "EvoRAG" / "results" / "entity_resolution_eval_report.json"
DEFAULT_OUTPUT_DIR = BACKEND_PYTHON_ROOT / "tests" / "EvoRAG" / "results"
DEFAULT_TOP_K = 5
DEFAULT_PREFLIGHT_RETRY_DELAYS = (2, 4, 8, 16)
DEFAULT_STRATEGY = "hybrid_rrf_baseline"
MEMORY_GUARDED_HYBRID_STRATEGY = "memory_guarded_hybrid"
LOW_SCORE_DIRECT_REJECT_THRESHOLD = 0.5
MANUAL_REVIEW_SCORE_MIN = 0.7
MANUAL_REVIEW_SCORE_MAX = 0.85
MANUAL_REVIEW_COMPONENT_SCORE_DIFF_MIN = 0.4
HIGH_SCORE_LLM_REJECT_MIN = 0.85
HYBRID_FLOW_STAGES = {"hybrid_rrf", "no_relation_memory_fallback_hybrid", "memory_judge_fallback_hybrid"}
MYSQL_MEMORY_FLOW_STAGES = {"mysql_relation_memory_judge", "llm_relation_guard_reject", "memory_judge_fallback_hybrid"}
NON_MERGE_RELATION_DECISIONS = {
    "attribute_of_entity",
    "section_title_of_entity",
    "process_step_of_entity",
    "metric_or_property",
}
ADMISSION_PASS_DECISIONS = {"entity", "valid_entity", "candidate_entity"}
ADMISSION_REJECT_DECISIONS = {"attribute", "section_title", "process_step", "metric_or_property", "not_entity"}

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

ENTITY_ADMISSION_GUARD_SYSTEM_PROMPT = """
你是 EvoRAG 的实体准入守门裁判。你会收到一个 incoming entity。目标是在进入实体关系经验和混合检索前，判断它本身是否值得作为实体消歧对象。

规则：
- 如果 incoming 表示独立知识实体、概念、方法、系统、算法或数据结构，decision=entity。
- 如果 incoming 只是属性、指标、章节标题、过程步骤，或明显不是实体，不能进入实体合并；输出 attribute、metric_or_property、section_title、process_step 或 not_entity。
- 如果信息不足但仍可能是实体，decision=entity，不要因为缺少历史关系而拒绝。

只输出 JSON，不要 Markdown，不要解释。

JSON 格式：
{
  "decision": "entity | attribute | section_title | process_step | metric_or_property | not_entity",
  "confidence": 0.0,
  "reason": "一句话理由"
}
""".strip()
ENTITY_RELATION_MEMORY_JUDGE_SYSTEM_PROMPT = """
你是 EvoRAG 的实体关系记忆裁判。你会收到一个 incoming entity，以及历史维护的白名单和黑名单关系。

目标：只根据这些历史关系判断 incoming entity 是否能直接等价到某个候选实体。

规则：
- 白名单表示过去确认过的同一实体或等价表述。
- 黑名单表示过去确认过的非同一实体、属性误导、版本差异或相关但不同。
- 黑名单优先级高于名称相似度；如果白名单和黑名单冲突，输出 ambiguous。
- 如果能确定等价，decision=matched，并填写 matched_entity_id。
- 如果 incoming 只是某个候选实体的属性、指标、章节标题或过程步骤，不能合并；输出 attribute_of_entity、metric_or_property、section_title_of_entity 或 process_step_of_entity，并填写对应 matched_entity_id。
- 如果只是相关但不是同一实体，decision=not_matched。
- 如果不能确定，decision=ambiguous，不要猜测。
- 不要引入候选列表之外的实体。

只输出 JSON，不要 Markdown，不要解释。

JSON 格式：
{
  "decision": "matched | not_matched | ambiguous | attribute_of_entity | section_title_of_entity | process_step_of_entity | metric_or_property",
  "matched_entity_id": 0,
  "confidence": 0.0,
  "reason": "一句话理由"
}
""".strip()


def load_benchmark(fixture_path: Path = DEFAULT_FIXTURE_PATH) -> dict[str, Any]:
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def preflight_index_connections(
    hybrid_index: Any,
    *,
    progress: Callable[[str], None] | None = None,
    sleep_fn: Callable[[float], None] = sleep,
    retry_delays: Sequence[int] = DEFAULT_PREFLIGHT_RETRY_DELAYS,
) -> None:
    total_attempts = len(retry_delays) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            hybrid_index.ensure_indexes()
            emit(progress, "连接评测索引成功")
            return
        except Exception as exc:
            if attempt == total_attempts:
                emit(progress, f"连接评测索引最终失败: {exc}")
                raise RuntimeError(f"评测索引连接失败，已重试 {len(retry_delays)} 次: {exc}") from exc

            delay_seconds = retry_delays[attempt - 1]
            emit(progress, f"连接评测索引失败，第 {attempt}/{total_attempts} 次尝试，{delay_seconds} 秒后重试: {exc}")
            sleep_fn(delay_seconds)


def get_entity_resolution_strategy(name: str) -> Callable[..., Any]:
    strategies: dict[str, Callable[..., Any]] = {
        DEFAULT_STRATEGY: run_hybrid_rrf_baseline_strategy,
        MEMORY_GUARDED_HYBRID_STRATEGY: run_memory_guarded_hybrid_strategy,
    }
    strategy = str(name or "").strip()
    if strategy in strategies:
        return strategies[strategy]
    available = ", ".join(sorted(strategies))
    raise ValueError(f"Unsupported entity resolution strategy: {name}. Available strategies: {available}")


async def run_hybrid_rrf_baseline_strategy(
    *,
    case: dict[str, Any],
    incoming: IncomingEntity,
    embedding_client: EvoRAGEmbeddingClient,
    hybrid_index: EntityHybridIndex,
    judge_client: EvoRAGLLMClient,
    config: EvoRAGSettings,
    top_k: int,
    id_maps: dict[str, dict[Any, Any]],
    relation_memory: Any | None = None,
) -> dict[str, Any]:
    incoming.embedding = await embedding_client.embed_text(entity_text_for_embedding(incoming))
    candidates = hybrid_index.search(incoming, top_k=top_k)
    retrieval = build_retrieval_report(candidates, id_maps=id_maps)

    checkpoint = judge_client.usage_checkpoint() if hasattr(judge_client, "usage_checkpoint") else 0
    judgment = await judge_entity_resolution_case(
        case,
        incoming=incoming,
        retrieval=retrieval,
        llm_client=judge_client,
        config=config,
    )
    case_judge_usage = judge_client.usage_records_since(checkpoint) if hasattr(judge_client, "usage_records_since") else []
    experience_record = record_hybrid_resolution_experience(
        incoming=incoming,
        candidates=candidates,
        judgment=judgment,
        relation_memory=relation_memory,
        id_maps=id_maps,
    )
    return {
        "retrieval": retrieval,
        "judgment": judgment,
        "judge_usage": case_judge_usage,
        "strategy_trace": {"strategy": DEFAULT_STRATEGY, "stage": "hybrid_rrf"},
        "experience": experience_record,
    }


async def run_memory_guarded_hybrid_strategy(
    *,
    case: dict[str, Any],
    incoming: IncomingEntity,
    embedding_client: EvoRAGEmbeddingClient,
    hybrid_index: EntityHybridIndex,
    judge_client: EvoRAGLLMClient,
    config: EvoRAGSettings,
    top_k: int,
    id_maps: dict[str, dict[Any, Any]],
    relation_memory: EntityRelationMemory | Any | None = None,
) -> dict[str, Any]:
    if relation_memory is None:
        relation_memory = build_default_relation_memory(config)

    direct_relation = relation_memory.get_direct_relation(incoming)
    if direct_relation is not None and is_allow_relation(direct_relation.decision):
        return build_relation_memory_match_result(
            direct_relation,
            id_maps=id_maps,
            trace_stage="redis_direct_allow",
            reason=f"redis direct relation matched: {direct_relation.reason}",
        )

    admission_checkpoint = judge_client.usage_checkpoint() if hasattr(judge_client, "usage_checkpoint") else 0
    admission_judgment = await judge_entity_admission_guard(
        case,
        incoming=incoming,
        llm_client=judge_client,
        config=config,
    )
    admission_usage = judge_client.usage_records_since(admission_checkpoint) if hasattr(judge_client, "usage_records_since") else []
    if is_admission_reject_decision(admission_judgment["decision"]):
        return build_admission_guard_reject_result(admission_judgment, judge_usage=admission_usage)

    relations = relation_memory.list_top_relations(incoming, limit=30)
    if relations:
        checkpoint = judge_client.usage_checkpoint() if hasattr(judge_client, "usage_checkpoint") else 0
        memory_judgment = await judge_relation_memory_case(
            case,
            incoming=incoming,
            relations=relations,
            llm_client=judge_client,
            config=config,
        )
        memory_usage = judge_client.usage_records_since(checkpoint) if hasattr(judge_client, "usage_records_since") else []
        matched_record = find_relation_record(relations, memory_judgment.get("matched_entity_id"))
        if is_non_merge_relation_decision(memory_judgment["decision"]):
            return build_relation_guard_reject_result(
                memory_judgment,
                matched_record,
                id_maps=id_maps,
                relation_memory=relation_memory,
                incoming=incoming,
                judge_usage=[*admission_usage, *memory_usage],
            )
        if memory_judgment["decision"] == "matched" and matched_record is not None:
            return build_relation_memory_match_result(
                matched_record,
                id_maps=id_maps,
                trace_stage="mysql_relation_memory_judge",
                reason=str(memory_judgment.get("reason") or matched_record.reason),
                confidence=float(memory_judgment.get("confidence") or matched_record.confidence),
                judge_usage=[*admission_usage, *memory_usage],
            )

        fallback = await run_hybrid_rrf_baseline_strategy(
            case=case,
            incoming=incoming,
            embedding_client=embedding_client,
            hybrid_index=hybrid_index,
            judge_client=judge_client,
            config=config,
            top_k=top_k,
            id_maps=id_maps,
            relation_memory=relation_memory,
        )
        fallback["judge_usage"] = [*admission_usage, *memory_usage, *fallback.get("judge_usage", [])]
        fallback["strategy_trace"] = {
            "strategy": MEMORY_GUARDED_HYBRID_STRATEGY,
            "stage": "memory_judge_fallback_hybrid",
            "memory_decision": memory_judgment["decision"],
            "relation_count": len(relations),
        }
        return fallback

    fallback = await run_hybrid_rrf_baseline_strategy(
        case=case,
        incoming=incoming,
        embedding_client=embedding_client,
        hybrid_index=hybrid_index,
        judge_client=judge_client,
        config=config,
        top_k=top_k,
        id_maps=id_maps,
        relation_memory=relation_memory,
    )
    fallback["judge_usage"] = [*admission_usage, *fallback.get("judge_usage", [])]
    fallback["strategy_trace"] = {
        "strategy": MEMORY_GUARDED_HYBRID_STRATEGY,
        "stage": "no_relation_memory_fallback_hybrid",
        "direct_reject": bool(direct_relation is not None and is_reject_relation(direct_relation.decision)),
    }
    return fallback


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
    rebuild_seeds: bool = False,
    strategy: str = DEFAULT_STRATEGY,
    relation_memory: Any | None = None,
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
    preflight_index_connections(hybrid_index, progress=progress)
    strategy_runner = get_entity_resolution_strategy(strategy)
    if relation_memory is None and strategy == MEMORY_GUARDED_HYBRID_STRATEGY:
        relation_memory = build_default_relation_memory(runtime_config)

    id_maps = build_seed_id_maps(seed_entities)
    stored_entities = build_stored_seed_entities(seed_entities, scope=scope, id_maps=id_maps)
    if rebuild_seeds:
        emit(progress, "收到 --rebuild-seeds，强制重建评测 seed 数据")
        should_write_seeds = True
    else:
        should_write_seeds = not seed_entities_exist(hybrid_index, stored_entities)
        if not should_write_seeds:
            emit(progress, f"评测 seed 已存在，跳过写入: {len(stored_entities)}/{len(stored_entities)}")

    if should_write_seeds:
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
        strategy_result = await strategy_runner(
            case=case,
            incoming=incoming,
            embedding_client=embedding_client,
            hybrid_index=hybrid_index,
            judge_client=judge_client,
            config=runtime_config,
            top_k=top_k,
            id_maps=id_maps,
            relation_memory=relation_memory,
        )
        retrieval = strategy_result["retrieval"]
        judgment = strategy_result["judgment"]
        case_judge_usage = strategy_result["judge_usage"]
        judge_usage_records.extend(case_judge_usage)
        case_report = {
            "case_id": case["case_id"],
            "category": case["category"],
            "incoming_entity": case["incoming_entity"],
            "expected": case["expected"],
            "retrieval": retrieval,
            "judgment": judgment,
            "runtime": {"total_case_ms": elapsed_ms(case_started_at)},
            "usage": {"judge": summarize_llm_usage(case_judge_usage)},
        }
        if strategy_result.get("strategy_trace"):
            case_report["strategy_trace"] = strategy_result["strategy_trace"]
        if strategy_result.get("experience"):
            case_report["experience"] = strategy_result["experience"]
        case_report["flow"] = build_case_flow(case_report)
        case_reports.append(case_report)

    report = {
        "dataset_id": benchmark.get("dataset_id", ""),
        "fixture_path": str(fixture_path),
        "output_path": str(output_path),
        "strategy": strategy,
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
        "flow_metrics": compute_flow_metrics(case_reports),
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


async def judge_entity_admission_guard(
    case: dict[str, Any],
    *,
    incoming: IncomingEntity,
    llm_client: EvoRAGLLMClient,
    config: EvoRAGSettings = settings,
) -> dict[str, Any]:
    data = await llm_client.chat_json(
        system_prompt=ENTITY_ADMISSION_GUARD_SYSTEM_PROMPT,
        user_payload={
            "case_id": case["case_id"],
            "incoming_entity": incoming_snapshot(incoming),
        },
        operation_name=f"EvoRAG entity admission guard {case['case_id']}",
        model=config.entity_admission_judge_model,
    )
    return normalize_admission_guard_judgment(data)

async def judge_relation_memory_case(
    case: dict[str, Any],
    *,
    incoming: IncomingEntity,
    relations: list[EntityRelationMemoryRecord],
    llm_client: EvoRAGLLMClient,
    config: EvoRAGSettings = settings,
) -> dict[str, Any]:
    allow_relations = [relation for relation in relations if is_allow_relation(relation.decision)]
    reject_relations = [relation for relation in relations if is_reject_relation(relation.decision)]
    data = await llm_client.chat_json(
        system_prompt=ENTITY_RELATION_MEMORY_JUDGE_SYSTEM_PROMPT,
        user_payload={
            "case_id": case["case_id"],
            "incoming_entity": incoming_snapshot(incoming),
            "allow_relations": [relation_memory_snapshot(relation) for relation in allow_relations],
            "reject_relations": [relation_memory_snapshot(relation) for relation in reject_relations],
        },
        operation_name=f"EvoRAG entity relation memory judge {case['case_id']}",
        model=config.entity_admission_judge_model,
    )
    return normalize_relation_memory_judgment(data)


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


def normalize_admission_guard_judgment(data: dict[str, Any]) -> dict[str, Any]:
    decision = str(data.get("decision") or "entity").strip().lower()
    if decision not in {*ADMISSION_PASS_DECISIONS, *ADMISSION_REJECT_DECISIONS}:
        decision = "entity"
    return {
        "decision": decision,
        "confidence": float(data.get("confidence") or 0.0),
        "reason": str(data.get("reason") or ""),
    }

def normalize_relation_memory_judgment(data: dict[str, Any]) -> dict[str, Any]:
    decision = str(data.get("decision") or "ambiguous").strip().lower()
    if decision not in {"matched", "not_matched", "ambiguous", *NON_MERGE_RELATION_DECISIONS}:
        decision = "ambiguous"
    matched_entity_id = data.get("matched_entity_id")
    return {
        "decision": decision,
        "matched_entity_id": int(matched_entity_id) if matched_entity_id else None,
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
        metrics[f"recall@{k}"] = nullable_rate(counts[f"matched_found@{k}"], counts["matched_cases"])
    metrics["mrr"] = nullable_rate(reciprocal_rank_sum, counts["matched_cases"])
    metrics["ambiguous_coverage@3"] = nullable_rate(counts["ambiguous_covered@3"], counts["ambiguous_cases"])
    metrics["close_negative_top1_rate"] = nullable_rate(counts["close_negative_top1"], counts["close_negative_cases"])
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
        "decision_accuracy": nullable_rate(counts["correct_decisions"], counts["total_cases"]),
        "matched_entity_accuracy": nullable_rate(counts["correct_matched_entities"], counts["expected_matched_cases"]),
        "false_match_rate": nullable_rate(counts["false_matches"], counts["expected_non_matched_cases"]),
        "false_new_rate": nullable_rate(counts["false_new_cases"], counts["expected_matched_cases"]),
        "false_ambiguous_rate": nullable_rate(counts["false_ambiguous_cases"], counts["clear_cases"]),
        "ambiguous_accuracy": nullable_rate(counts["correct_ambiguous_cases"], counts["ambiguous_cases"]),
        "counts": counts,
    }


def is_case_resolution_correct(report: dict[str, Any]) -> bool:
    expected = report.get("expected", {})
    judgment = report.get("judgment", {})
    expected_decision = str(expected.get("decision") or "")
    actual_decision = str(judgment.get("decision") or "")
    if expected_decision == "matched":
        return actual_decision == "matched" and judgment.get("matched_seed_id") == expected.get("matched_seed_id")
    return actual_decision == expected_decision


def build_case_flow(report: dict[str, Any]) -> dict[str, Any]:
    trace = report.get("strategy_trace") or {}
    stage = str(trace.get("stage") or "unknown")
    strategy = str(trace.get("strategy") or "")
    experience = report.get("experience") or {}
    used_memory_guarded = strategy == MEMORY_GUARDED_HYBRID_STRATEGY
    wrote_experience = bool(experience.get("recorded"))
    manual_review_triggered = str(experience.get("source") or "") == "manual"
    return {
        "stage": stage,
        "used_redis": used_memory_guarded,
        "used_admission_guard": used_memory_guarded and stage != "redis_direct_allow",
        "used_mysql_memory": stage in MYSQL_MEMORY_FLOW_STAGES,
        "used_relation_guard": stage in MYSQL_MEMORY_FLOW_STAGES,
        "used_hybrid": stage in HYBRID_FLOW_STAGES,
        "wrote_experience": wrote_experience,
        "manual_review_triggered": manual_review_triggered,
        "experience_source": experience.get("source") if wrote_experience else None,
        "experience_relation_type": experience.get("relation_type") if wrote_experience else None,
    }


def compute_flow_metrics(case_reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, int] = defaultdict(int)
    by_category_stage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stage_accuracy_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    category_stage_accuracy_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    component_usage_counts = {
        "redis": 0,
        "admission_guard": 0,
        "mysql_memory": 0,
        "relation_guard": 0,
        "hybrid": 0,
        "experience_write": 0,
        "manual_review_trigger": 0,
    }

    for report in case_reports:
        flow = report.get("flow") or build_case_flow(report)
        stage = str(flow.get("stage") or "unknown")
        category = str(report.get("category") or "")
        by_stage[stage] += 1
        by_category_stage[category][stage] += 1

        correct = is_case_resolution_correct(report)
        stage_accuracy_counts[stage]["total"] += 1
        category_stage_accuracy_counts[category][stage]["total"] += 1
        if correct:
            stage_accuracy_counts[stage]["correct"] += 1
            category_stage_accuracy_counts[category][stage]["correct"] += 1

        if flow.get("used_redis"):
            component_usage_counts["redis"] += 1
        if flow.get("used_admission_guard"):
            component_usage_counts["admission_guard"] += 1
        if flow.get("used_mysql_memory"):
            component_usage_counts["mysql_memory"] += 1
        if flow.get("used_relation_guard"):
            component_usage_counts["relation_guard"] += 1
        if flow.get("used_hybrid"):
            component_usage_counts["hybrid"] += 1
        if flow.get("wrote_experience"):
            component_usage_counts["experience_write"] += 1
        if flow.get("manual_review_triggered"):
            component_usage_counts["manual_review_trigger"] += 1

    accuracy_by_stage = {
        stage: {
            "correct": counts["correct"],
            "total": counts["total"],
            "accuracy": safe_rate(counts["correct"], counts["total"]),
        }
        for stage, counts in stage_accuracy_counts.items()
    }
    accuracy_by_category_stage = {
        category: {
            stage: {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": safe_rate(counts["correct"], counts["total"]),
            }
            for stage, counts in stage_counts.items()
        }
        for category, stage_counts in category_stage_accuracy_counts.items()
    }

    return {
        "total_cases": len(case_reports),
        "by_stage": dict(by_stage),
        "by_category_stage": {category: dict(stage_counts) for category, stage_counts in by_category_stage.items()},
        "accuracy_by_stage": accuracy_by_stage,
        "accuracy_by_category_stage": accuracy_by_category_stage,
        "component_usage_counts": component_usage_counts,
        "hybrid_fallback_rate": safe_rate(component_usage_counts["hybrid"], len(case_reports)),
        "experience_write_count": component_usage_counts["experience_write"],
        "manual_review_trigger_count": component_usage_counts["manual_review_trigger"],
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


def seed_entities_exist(hybrid_index: Any, stored_entities: Sequence[StoredEntity]) -> bool:
    if not stored_entities:
        return True
    checker = getattr(hybrid_index, "seed_entities_exist", None)
    if checker is None:
        return False
    return bool(checker(stored_entities))


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


def is_admission_reject_decision(decision: str) -> bool:
    return str(decision or "").strip().lower() in ADMISSION_REJECT_DECISIONS


def build_admission_guard_reject_result(
    judgment: dict[str, Any],
    *,
    judge_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    decision = str(judgment.get("decision") or "not_entity")
    confidence = float(judgment.get("confidence") or 0.0)
    reason = str(judgment.get("reason") or "admission guard rejected non-entity incoming")
    return {
        "retrieval": {"ranked_seed_ids": [], "candidates": []},
        "judgment": {
            "decision": "new",
            "matched_seed_id": None,
            "confidence": confidence,
            "reason": reason,
        },
        "judge_usage": judge_usage or [],
        "strategy_trace": {
            "strategy": MEMORY_GUARDED_HYBRID_STRATEGY,
            "stage": "llm_admission_guard_reject",
            "admission_decision": decision,
        },
    }

def is_non_merge_relation_decision(decision: str) -> bool:
    return str(decision or "").strip().lower() in NON_MERGE_RELATION_DECISIONS


def build_relation_guard_reject_result(
    judgment: dict[str, Any],
    relation: EntityRelationMemoryRecord | None,
    *,
    id_maps: dict[str, dict[Any, Any]],
    relation_memory: Any | None,
    incoming: IncomingEntity,
    judge_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    decision = str(judgment.get("decision") or "not_matched")
    confidence = float(judgment.get("confidence") or 0.0)
    reason = str(judgment.get("reason") or "relation guard rejected entity merge")
    candidates = []
    experience_record = None
    if relation is not None:
        candidate = CandidateEntity(
            entity=relation.candidate,
            score=confidence,
            rank=1,
            source="relation_guard",
            vector_score=0.0,
            es_score=0.0,
        )
        candidates.append(candidate)
        writer = getattr(relation_memory, "record_relation", None) if relation_memory is not None else None
        if writer is not None:
            experience_record = write_relation_experience(
                writer,
                incoming=incoming,
                candidate=relation.candidate,
                decision="reject",
                relation_type=decision,
                confidence=confidence,
                source="llm_guard",
                reason=reason,
            )

    result = {
        "retrieval": build_retrieval_report(candidates, id_maps=id_maps),
        "judgment": {
            "decision": "new",
            "matched_seed_id": None,
            "confidence": confidence,
            "reason": reason,
        },
        "judge_usage": judge_usage or [],
        "strategy_trace": {
            "strategy": MEMORY_GUARDED_HYBRID_STRATEGY,
            "stage": "llm_relation_guard_reject",
            "relation_guard_decision": decision,
            "matched_entity_id": relation.candidate.id if relation is not None else None,
        },
    }
    if experience_record is not None:
        result["experience"] = experience_record
    return result

def record_hybrid_resolution_experience(
    *,
    incoming: IncomingEntity,
    candidates: list[CandidateEntity],
    judgment: dict[str, Any],
    relation_memory: Any | None,
    id_maps: dict[str, dict[Any, Any]],
) -> dict[str, Any] | None:
    if relation_memory is None or not candidates:
        return None
    writer = getattr(relation_memory, "record_relation", None)
    if writer is None:
        return None

    best = candidates[0]
    if best.score < LOW_SCORE_DIRECT_REJECT_THRESHOLD:
        return write_relation_experience(
            writer,
            incoming=incoming,
            candidate=best.entity,
            decision="reject",
            relation_type="low_score_direct_reject",
            confidence=round(1.0 - float(best.score), 6),
            source="auto",
            reason=str(judgment.get("reason") or "hybrid score below direct reject threshold"),
        )

    if judgment.get("decision") == "new" and best.score >= HIGH_SCORE_LLM_REJECT_MIN:
        return write_relation_experience(
            writer,
            incoming=incoming,
            candidate=best.entity,
            decision="reject",
            relation_type="high_score_llm_reject",
            confidence=1.0,
            source="llm_judge",
            reason=str(judgment.get("reason") or "final judge rejected a high-score candidate"),
        )

    score_requires_manual_review = MANUAL_REVIEW_SCORE_MIN <= best.score <= MANUAL_REVIEW_SCORE_MAX
    component_score_delta = abs(float(best.es_score or 0.0) - float(best.vector_score or 0.0))
    components_require_manual_review = component_score_delta > MANUAL_REVIEW_COMPONENT_SCORE_DIFF_MIN
    if not (score_requires_manual_review or components_require_manual_review):
        return None

    if judgment.get("decision") == "matched":
        matched_candidate = candidate_from_judgment(candidates, judgment, id_maps=id_maps)
        if matched_candidate is None:
            return None
        return write_relation_experience(
            writer,
            incoming=incoming,
            candidate=matched_candidate.entity,
            decision="allow",
            relation_type="manual_match",
            confidence=1.0,
            source="manual",
            reason=str(judgment.get("reason") or "manual review matched"),
        )

    return write_relation_experience(
        writer,
        incoming=incoming,
        candidate=best.entity,
        decision="reject",
        relation_type="manual_reject",
        confidence=1.0,
        source="manual",
        reason=str(judgment.get("reason") or "manual review rejected"),
    )


def write_relation_experience(
    writer: Callable[..., Any],
    *,
    incoming: IncomingEntity,
    candidate: StoredEntity,
    decision: str,
    relation_type: str,
    confidence: float,
    source: str,
    reason: str,
) -> dict[str, Any]:
    writer(
        incoming=incoming,
        candidate=candidate,
        decision=decision,
        relation_type=relation_type,
        confidence=confidence,
        source=source,
        reason=reason,
    )
    return {
        "recorded": True,
        "candidate_entity_id": candidate.id,
        "decision": decision,
        "relation_type": relation_type,
        "confidence": confidence,
        "source": source,
    }


def candidate_from_judgment(
    candidates: list[CandidateEntity],
    judgment: dict[str, Any],
    *,
    id_maps: dict[str, dict[Any, Any]],
) -> CandidateEntity | None:
    matched_seed_id = judgment.get("matched_seed_id")
    if matched_seed_id:
        entity_id = id_maps["numeric_id_by_seed_id"].get(str(matched_seed_id))
        if entity_id is not None:
            return next((candidate for candidate in candidates if candidate.entity.id == int(entity_id)), None)
    return None


def build_relation_memory_match_result(
    relation: EntityRelationMemoryRecord,
    *,
    id_maps: dict[str, dict[Any, Any]],
    trace_stage: str,
    reason: str,
    confidence: float | None = None,
    judge_usage: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    score = float(confidence if confidence is not None else relation.confidence)
    candidate = CandidateEntity(
        entity=relation.candidate,
        score=score,
        rank=1,
        source="relation_memory",
        vector_score=0.0,
        es_score=0.0,
    )
    retrieval = build_retrieval_report([candidate], id_maps=id_maps)
    matched_seed_id = retrieval["ranked_seed_ids"][0] if retrieval["ranked_seed_ids"] else None
    return {
        "retrieval": retrieval,
        "judgment": {
            "decision": "matched",
            "matched_seed_id": matched_seed_id,
            "confidence": score,
            "reason": reason,
        },
        "judge_usage": judge_usage or [],
        "strategy_trace": {
            "strategy": MEMORY_GUARDED_HYBRID_STRATEGY,
            "stage": trace_stage,
            "relation_id": relation.id,
            "relation_decision": relation.decision,
            "matched_entity_id": relation.candidate.id,
        },
    }


def find_relation_record(
    relations: list[EntityRelationMemoryRecord],
    matched_entity_id: Any,
) -> EntityRelationMemoryRecord | None:
    if matched_entity_id is None:
        return None
    try:
        entity_id = int(matched_entity_id)
    except (TypeError, ValueError):
        return None
    return next((relation for relation in relations if relation.candidate.id == entity_id), None)


def relation_memory_snapshot(relation: EntityRelationMemoryRecord) -> dict[str, Any]:
    entity = relation.candidate
    return {
        "relation_id": relation.id,
        "decision": relation.decision,
        "relation_type": relation.relation_type,
        "confidence": relation.confidence,
        "hit_count": relation.hit_count,
        "source": relation.source,
        "reason": relation.reason,
        "candidate_entity": {
            "id": entity.id,
            "canonical_name": entity.canonical_name,
            "normalized_name": entity.normalized_name,
            "entity_type": entity.entity_type,
            "aliases": entity.aliases,
            "identity_description": entity.identity_description,
            "summary": entity.summary,
            "description_for_match": entity.description_for_match,
        },
    }


def build_default_relation_memory(config: EvoRAGSettings) -> EntityRelationMemory:
    memory = EntityRelationMemory(MySQLEntityRepository(config))
    memory.init_schema()
    return memory


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


def nullable_rate(numerator: float, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(float(numerator) / denominator, 6)


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def safe_path_part(value: Any) -> str:
    text = str(value).strip() if value is not None else "all"
    if not text:
        return "all"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in text)


def build_parameterized_output_path(
    *,
    strategy: str,
    category: str | None,
    limit: int | None,
    top_k: int,
    rebuild_seeds: bool,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    parts = [
        safe_path_part(strategy),
        f"category-{safe_path_part(category)}",
        f"limit-{safe_path_part(limit)}",
        f"top-k-{safe_path_part(top_k)}",
    ]
    if rebuild_seeds:
        parts.append("rebuild-seeds")
    run_signature = "__".join(parts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / run_signature / f"entity_resolution_eval_report__{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EvoRAG entity-resolution retrieval + DeepSeek judge evaluation.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE_PATH)
    parser.add_argument("--out", type=Path, default=None, help="Explicit report output path. Defaults to a parameterized run path under tests/EvoRAG/results.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--rebuild-seeds", action="store_true", help="Force rebuilding eval seed entities in ES and Milvus.")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, help="Entity resolution strategy to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.out or build_parameterized_output_path(
        strategy=args.strategy,
        category=args.category,
        limit=args.limit,
        top_k=args.top_k,
        rebuild_seeds=args.rebuild_seeds,
    )

    def print_progress(message: str) -> None:
        print(message, flush=True)

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=args.fixture,
            output_path=output_path,
            top_k=args.top_k,
            limit=args.limit,
            category=args.category,
            rebuild_seeds=args.rebuild_seeds,
            strategy=args.strategy,
            progress=print_progress,
        )
    )
    print(json.dumps({"output_path": str(output_path), "strategy": report["strategy"], "overall": report["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()






