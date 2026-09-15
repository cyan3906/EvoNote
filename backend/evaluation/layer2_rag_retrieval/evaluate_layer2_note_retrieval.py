from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_QUERIES_PATH = Path("backend/testdata/408_note_retrieval_queries_fixture.json")
DEFAULT_OUTPUT_PATH = Path("backend/evaluation/results/layer2_note_retrieval_metrics.json")
DEFAULT_K_VALUES = (1, 3, 5, 10)
CHANNELS = ("es", "milvus", "rrf")


@dataclass(frozen=True)
class RelevantNote:
    note_id: str
    relevance: int


@dataclass(frozen=True)
class RetrievalQuery:
    query_id: str
    query_text: str
    keywords: list[str]
    gold_relevant_notes: list[RelevantNote]
    exclude_note_ids: set[str]


@dataclass(frozen=True)
class RetrievalResult:
    note_id: str
    score: float | None = None
    title: str = ""
    reason: str = ""


def evaluate_note_retrieval(
    queries_path: Path,
    output_path: Path,
    *,
    results_path: Path | None = None,
    live: bool = False,
    smoke_oracle: bool = False,
    top_k: int = 10,
) -> dict[str, Any]:
    queries = load_queries(queries_path)

    if live:
        retrieval_results, errors = run_live_retrieval(queries, top_k=top_k)
        input_mode = "live"
    elif results_path is not None:
        retrieval_results = load_results(results_path)
        errors = []
        input_mode = "results_file"
    elif smoke_oracle:
        retrieval_results = build_oracle_results(queries)
        errors = []
        input_mode = "oracle_smoke"
    else:
        retrieval_results = {}
        errors = ["No retrieval results were provided. Pass --results, --live, or --smoke-oracle."]
        input_mode = "not_run"

    metrics = {channel: compute_channel_metrics(queries, retrieval_results, channel) for channel in CHANNELS}
    per_query = [build_query_report(query, retrieval_results) for query in queries]
    report = {
        "layer": "layer2_rag_retrieval",
        "target_level": "note",
        "input_mode": input_mode,
        "queries_path": str(queries_path),
        "results_path": str(results_path) if results_path else None,
        "top_k": top_k,
        "metrics": metrics,
        "channel_comparison": compute_channel_comparison(queries, retrieval_results, k=top_k),
        "counts": {
            "queries": len(queries),
            "gold_relevant_note_refs": sum(len(query.gold_relevant_notes) for query in queries),
            "errors": len(errors),
        },
        "errors": errors,
        "queries": per_query,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def load_queries(path: Path) -> list[RetrievalQuery]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    queries: list[RetrievalQuery] = []

    for raw_query in data.get("queries", []):
        gold_notes = [
            RelevantNote(note_id=str(item["note_id"]), relevance=max(1, int(item.get("relevance", 1))))
            for item in raw_query.get("gold_relevant_notes", [])
            if item.get("note_id")
        ]
        queries.append(
            RetrievalQuery(
                query_id=str(raw_query["query_id"]),
                query_text=str(raw_query["query_text"]),
                keywords=[str(item) for item in raw_query.get("keywords", []) if str(item).strip()],
                gold_relevant_notes=gold_notes,
                exclude_note_ids={str(item) for item in raw_query.get("exclude_note_ids", [])},
            )
        )

    if not queries:
        raise ValueError(f"No queries found in {path}")

    return queries


def load_results(path: Path) -> dict[str, dict[str, list[RetrievalResult]]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    raw_queries = data.get("queries", data if isinstance(data, list) else [])
    results: dict[str, dict[str, list[RetrievalResult]]] = {}

    for raw_query in raw_queries:
        query_id = str(raw_query.get("query_id", ""))
        raw_results = raw_query.get("results", raw_query)
        if not query_id or not isinstance(raw_results, dict):
            continue

        results[query_id] = {}
        for channel in CHANNELS:
            results[query_id][channel] = normalize_result_list(raw_results.get(channel, []))

    return results


def normalize_result_list(raw_items: Any) -> list[RetrievalResult]:
    if not isinstance(raw_items, list):
        return []

    results: list[RetrievalResult] = []
    seen: set[str] = set()

    for raw_item in raw_items:
        if isinstance(raw_item, str):
            note_id = raw_item
            score = None
            title = ""
            reason = ""
        elif isinstance(raw_item, dict):
            note_id = str(raw_item.get("note_id") or raw_item.get("id") or "")
            score = parse_optional_float(raw_item.get("score"))
            title = str(raw_item.get("title", ""))
            reason = str(raw_item.get("reason", ""))
        else:
            continue

        if not note_id or note_id in seen:
            continue

        seen.add(note_id)
        results.append(RetrievalResult(note_id=note_id, score=score, title=title, reason=reason))

    return results


def run_live_retrieval(
    queries: list[RetrievalQuery],
    *,
    top_k: int,
) -> tuple[dict[str, dict[str, list[RetrievalResult]]], list[str]]:
    repo_root = Path(__file__).resolve().parents[3]
    backend_python = repo_root / "backend" / "python"
    if str(backend_python) not in sys.path:
        sys.path.insert(0, str(backend_python))

    errors: list[str] = []
    results: dict[str, dict[str, list[RetrievalResult]]] = {}

    try:
        from app.core import retrieval
    except Exception as exc:  # pragma: no cover - depends on local services.
        return {}, [f"Failed to import retrieval module: {exc}"]

    for query in queries:
        query_results: dict[str, list[RetrievalResult]] = {channel: [] for channel in CHANNELS}
        query_vector: list[float] = []

        try:
            query_vector = retrieval.safe_embed_text(build_query_text(query))
        except Exception as exc:
            errors.append(f"{query.query_id}: embedding failed: {exc}")

        try:
            hits = retrieval.query_elasticsearch_notes_by_l3(
                query.query_text,
                query.keywords,
                top_k=top_k,
                exclude_note_id=single_excluded_note(query),
            )
            query_results["es"] = [result_from_hit(hit) for hit in hits]
        except Exception as exc:
            errors.append(f"{query.query_id}: es retrieval failed: {exc}")

        try:
            hits = retrieval.query_milvus_notes_by_l2_vector(
                query_vector,
                top_k=top_k,
                exclude_note_id=single_excluded_note(query),
            )
            query_results["milvus"] = [result_from_hit(hit) for hit in hits]
        except Exception as exc:
            errors.append(f"{query.query_id}: milvus retrieval failed: {exc}")

        try:
            representation = {
                "l2_summary": query.query_text,
                "l3_text": build_query_text(query),
                "keywords": query.keywords,
                "vector": query_vector,
            }
            hits = retrieval.hybrid_retrieve_notes(
                representation,
                top_k=top_k,
                exclude_note_id=single_excluded_note(query),
            )
            query_results["rrf"] = [result_from_hit(hit) for hit in hits]
        except Exception as exc:
            errors.append(f"{query.query_id}: rrf retrieval failed: {exc}")

        results[query.query_id] = query_results

    return results, errors


def result_from_hit(hit: Any) -> RetrievalResult:
    return RetrievalResult(
        note_id=str(getattr(hit, "note_id", "")),
        score=parse_optional_float(getattr(hit, "score", None)),
        title=str(getattr(hit, "title", "")),
        reason=str(getattr(hit, "reason", "")),
    )


def build_oracle_results(queries: list[RetrievalQuery]) -> dict[str, dict[str, list[RetrievalResult]]]:
    results: dict[str, dict[str, list[RetrievalResult]]] = {}

    for query in queries:
        ranked = [
            RetrievalResult(note_id=item.note_id, score=float(item.relevance))
            for item in sorted(query.gold_relevant_notes, key=lambda item: item.relevance, reverse=True)
        ]
        results[query.query_id] = {channel: ranked for channel in CHANNELS}

    return results


def compute_channel_metrics(
    queries: list[RetrievalQuery],
    retrieval_results: dict[str, dict[str, list[RetrievalResult]]],
    channel: str,
) -> dict[str, float]:
    metric_totals: dict[str, float] = {}

    for k in DEFAULT_K_VALUES:
        metric_totals[f"recall_at_{k}"] = 0.0
        metric_totals[f"precision_at_{k}"] = 0.0
        metric_totals[f"hit_rate_at_{k}"] = 0.0

    metric_totals["mrr"] = 0.0
    metric_totals["ndcg_at_5"] = 0.0
    metric_totals["ndcg_at_10"] = 0.0

    for query in queries:
        ranked_note_ids = get_ranked_note_ids(retrieval_results, query.query_id, channel)
        gold = {item.note_id: item.relevance for item in query.gold_relevant_notes}

        for k in DEFAULT_K_VALUES:
            hits = count_hits(ranked_note_ids[:k], gold)
            metric_totals[f"recall_at_{k}"] += safe_divide_raw(hits, len(gold))
            metric_totals[f"precision_at_{k}"] += safe_divide_raw(hits, k)
            metric_totals[f"hit_rate_at_{k}"] += 1.0 if hits > 0 else 0.0

        metric_totals["mrr"] += reciprocal_rank(ranked_note_ids, gold)
        metric_totals["ndcg_at_5"] += ndcg_at_k(ranked_note_ids, gold, 5)
        metric_totals["ndcg_at_10"] += ndcg_at_k(ranked_note_ids, gold, 10)

    query_count = len(queries)
    return {key: round(value / query_count, 6) for key, value in metric_totals.items()}


def build_query_report(
    query: RetrievalQuery,
    retrieval_results: dict[str, dict[str, list[RetrievalResult]]],
) -> dict[str, Any]:
    gold = {item.note_id: item.relevance for item in query.gold_relevant_notes}
    report: dict[str, Any] = {
        "query_id": query.query_id,
        "query_text": query.query_text,
        "gold_relevant_notes": [
            {"note_id": item.note_id, "relevance": item.relevance}
            for item in query.gold_relevant_notes
        ],
        "channels": {},
    }

    for channel in CHANNELS:
        results = retrieval_results.get(query.query_id, {}).get(channel, [])
        ranked_note_ids = [item.note_id for item in results]
        report["channels"][channel] = {
            "retrieved_note_ids": ranked_note_ids,
            "first_relevant_rank": first_relevant_rank(ranked_note_ids, gold),
            "hit_at_10": count_hits(ranked_note_ids[:10], gold) > 0,
            "missed_relevant_note_ids_at_10": [
                note_id for note_id in gold if note_id not in set(ranked_note_ids[:10])
            ],
            "irrelevant_note_ids_at_10": [
                note_id for note_id in ranked_note_ids[:10] if note_id not in gold
            ],
        }

    return report


def compute_channel_comparison(
    queries: list[RetrievalQuery],
    retrieval_results: dict[str, dict[str, list[RetrievalResult]]],
    *,
    k: int,
) -> dict[str, int]:
    comparison = {
        f"es_only_hit_at_{k}": 0,
        f"milvus_only_hit_at_{k}": 0,
        f"both_hit_at_{k}": 0,
        f"neither_hit_at_{k}": 0,
        "rrf_improved_over_best_single_mrr_queries": 0,
        "rrf_degraded_below_best_single_mrr_queries": 0,
    }

    for query in queries:
        gold = {item.note_id: item.relevance for item in query.gold_relevant_notes}
        es_ranked = get_ranked_note_ids(retrieval_results, query.query_id, "es")
        milvus_ranked = get_ranked_note_ids(retrieval_results, query.query_id, "milvus")
        rrf_ranked = get_ranked_note_ids(retrieval_results, query.query_id, "rrf")
        es_rr = reciprocal_rank(es_ranked, gold)
        milvus_rr = reciprocal_rank(milvus_ranked, gold)
        rrf_rr = reciprocal_rank(rrf_ranked, gold)
        es_hit = count_hits(es_ranked[:k], gold) > 0
        milvus_hit = count_hits(milvus_ranked[:k], gold) > 0

        if es_hit and milvus_hit:
            comparison[f"both_hit_at_{k}"] += 1
        elif es_hit:
            comparison[f"es_only_hit_at_{k}"] += 1
        elif milvus_hit:
            comparison[f"milvus_only_hit_at_{k}"] += 1
        else:
            comparison[f"neither_hit_at_{k}"] += 1

        best_single_rr = max(es_rr, milvus_rr)
        if rrf_rr > best_single_rr:
            comparison["rrf_improved_over_best_single_mrr_queries"] += 1
        elif rrf_rr < best_single_rr:
            comparison["rrf_degraded_below_best_single_mrr_queries"] += 1

    return comparison


def get_ranked_note_ids(
    retrieval_results: dict[str, dict[str, list[RetrievalResult]]],
    query_id: str,
    channel: str,
) -> list[str]:
    return [item.note_id for item in retrieval_results.get(query_id, {}).get(channel, [])]


def count_hits(ranked_note_ids: list[str], gold: dict[str, int]) -> int:
    return len(set(ranked_note_ids) & set(gold))


def first_relevant_rank(ranked_note_ids: list[str], gold: dict[str, int]) -> int | None:
    for rank, note_id in enumerate(ranked_note_ids, start=1):
        if note_id in gold:
            return rank
    return None


def reciprocal_rank(ranked_note_ids: list[str], gold: dict[str, int]) -> float:
    rank = first_relevant_rank(ranked_note_ids, gold)
    return 0.0 if rank is None else 1.0 / rank


def ndcg_at_k(ranked_note_ids: list[str], gold: dict[str, int], k: int) -> float:
    gains = [gold.get(note_id, 0) for note_id in ranked_note_ids[:k]]
    dcg = discounted_cumulative_gain(gains)
    ideal_gains = sorted(gold.values(), reverse=True)[:k]
    idcg = discounted_cumulative_gain(ideal_gains)
    return safe_divide_raw(dcg, idcg)


def discounted_cumulative_gain(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))


def safe_divide_raw(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_query_text(query: RetrievalQuery) -> str:
    return " ".join([query.query_text, *query.keywords]).strip()


def single_excluded_note(query: RetrievalQuery) -> str | None:
    if len(query.exclude_note_ids) == 1:
        return next(iter(query.exclude_note_ids))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate note-level ES/Milvus/RRF retrieval quality.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH, help="Retrieval query fixture JSON.")
    parser.add_argument("--results", type=Path, default=None, help="Offline retrieval results JSON.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output metrics JSON path.")
    parser.add_argument("--top-k", type=int, default=10, help="Top K to request in live mode and compare channels.")
    parser.add_argument("--live", action="store_true", help="Run live ES/Milvus/RRF retrieval through backend code.")
    parser.add_argument("--smoke-oracle", action="store_true", help="Use gold notes as retrieved results for a metrics smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_note_retrieval(
        args.queries,
        args.out,
        results_path=args.results,
        live=args.live,
        smoke_oracle=args.smoke_oracle,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "input_mode": report["input_mode"],
                "metrics": report["metrics"],
                "errors": report["errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
