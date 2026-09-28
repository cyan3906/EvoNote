import asyncio
import json
from pathlib import Path

from app.EvoRAG.entity_resolution_eval import (
    compute_deepseek_decision_metrics,
    compute_retrieval_metrics,
    evaluate_entity_resolution_suite,
    load_benchmark,
)


def test_load_benchmark_preserves_unified_scope_and_counts() -> None:
    benchmark = load_benchmark()

    assert benchmark["dataset_id"] == "entity-resolution-benchmark-v1"
    assert benchmark["scope"]["collection_id"] == "benchmark-v1"
    assert len(benchmark["seed_entities"]) == 70
    assert len(benchmark["cases"]) == 60


def test_compute_retrieval_metrics_for_ranked_candidates() -> None:
    case_reports = [
        {
            "category": "exact_match",
            "expected": {"decision": "matched", "matched_seed_id": "seed-001", "must_not_match_seed_ids": []},
            "retrieval": {"ranked_seed_ids": ["seed-001", "seed-002"]},
        },
        {
            "category": "paraphrase_match",
            "expected": {"decision": "matched", "matched_seed_id": "seed-003", "must_not_match_seed_ids": []},
            "retrieval": {"ranked_seed_ids": ["seed-002", "seed-004", "seed-003"]},
        },
        {
            "category": "related_but_different",
            "expected": {"decision": "matched", "matched_seed_id": "seed-005", "must_not_match_seed_ids": ["seed-006"]},
            "retrieval": {"ranked_seed_ids": ["seed-006", "seed-005"]},
        },
        {
            "category": "ambiguous_entity",
            "expected": {
                "decision": "ambiguous",
                "ambiguous_seed_ids": ["seed-007", "seed-008"],
                "must_not_match_seed_ids": [],
            },
            "retrieval": {"ranked_seed_ids": ["seed-008", "seed-007", "seed-009"]},
        },
        {
            "category": "nil_new_entity",
            "expected": {"decision": "new", "matched_seed_id": None, "must_not_match_seed_ids": ["seed-010"]},
            "retrieval": {"ranked_seed_ids": ["seed-010", "seed-011"]},
        },
    ]

    metrics = compute_retrieval_metrics(case_reports, k_values=(1, 3, 5))

    assert metrics["recall@1"] == 0.333333
    assert metrics["recall@3"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["mrr"] == 0.611111
    assert metrics["ambiguous_coverage@3"] == 1.0
    assert metrics["close_negative_top1_rate"] == 1.0
    assert metrics["counts"] == {
        "matched_cases": 3,
        "matched_found@1": 1,
        "matched_found@3": 3,
        "matched_found@5": 3,
        "ambiguous_cases": 1,
        "ambiguous_covered@3": 1,
        "close_negative_cases": 2,
        "close_negative_top1": 2,
    }


def test_compute_deepseek_decision_metrics_from_judgments() -> None:
    case_reports = [
        {
            "category": "exact_match",
            "expected": {"decision": "matched", "matched_seed_id": "seed-001"},
            "judgment": {"decision": "matched", "matched_seed_id": "seed-001"},
        },
        {
            "category": "alias_match",
            "expected": {"decision": "matched", "matched_seed_id": "seed-002"},
            "judgment": {"decision": "matched", "matched_seed_id": "seed-999"},
        },
        {
            "category": "ambiguous_entity",
            "expected": {"decision": "ambiguous", "matched_seed_id": None},
            "judgment": {"decision": "matched", "matched_seed_id": "seed-003"},
        },
        {
            "category": "nil_new_entity",
            "expected": {"decision": "new", "matched_seed_id": None},
            "judgment": {"decision": "ambiguous", "matched_seed_id": None},
        },
    ]

    metrics = compute_deepseek_decision_metrics(case_reports)

    assert metrics == {
        "decision_accuracy": 0.5,
        "matched_entity_accuracy": 0.5,
        "false_match_rate": 0.5,
        "false_new_rate": 0.0,
        "false_ambiguous_rate": 0.333333,
        "ambiguous_accuracy": 0.0,
        "counts": {
            "total_cases": 4,
            "correct_decisions": 2,
            "expected_matched_cases": 2,
            "correct_matched_entities": 1,
            "expected_non_matched_cases": 2,
            "false_matches": 1,
            "false_new_cases": 0,
            "clear_cases": 3,
            "false_ambiguous_cases": 1,
            "ambiguous_cases": 1,
            "correct_ambiguous_cases": 0,
        },
    }


def test_evaluate_entity_resolution_suite_uses_fakes_and_emits_progress(tmp_path: Path) -> None:
    fixture_path = tmp_path / "benchmark.json"
    output_path = tmp_path / "report.json"
    fixture_path.write_text(
        json.dumps(
            {
                "dataset_id": "tiny",
                "scope": {
                    "workspace_id": "eval",
                    "project_id": "entity-resolution",
                    "collection_id": "tiny",
                    "domain": "computer-science",
                },
                "seed_entities": [
                    {
                        "seed_id": "seed-001",
                        "canonical_name": "MVCC",
                        "entity_type": "concept",
                        "aliases": ["多版本并发控制"],
                        "identity_description": "数据库多版本并发控制机制。",
                    }
                ],
                "cases": [
                    {
                        "case_id": "case-001",
                        "category": "alias_match",
                        "incoming_entity": {
                            "name": "多版本并发控制",
                            "entity_type": "concept",
                            "aliases": [],
                            "identity_description": "减少读写阻塞的并发控制机制。",
                        },
                        "expected": {
                            "decision": "matched",
                            "matched_seed_id": "seed-001",
                            "expected_top_candidates": ["seed-001"],
                            "must_not_match_seed_ids": [],
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    progress: list[str] = []

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=FakeHybridIndex(),
            judge_client=FakeJudgeClient(),
            progress=progress.append,
        )
    )

    assert output_path.exists()
    assert any("加载数据集 tiny" in item for item in progress)
    assert any("写入 seed 实体 1/1: seed-001 MVCC" in item for item in progress)
    assert any("处理 case 1/1: case-001 alias_match" in item for item in progress)
    assert report["total_cases"] == 1
    assert report["overall"]["retrieval"]["recall@1"] == 1.0
    assert report["overall"]["deepseek_judge"]["decision_accuracy"] == 1.0


class FakeEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeHybridIndex:
    def __init__(self) -> None:
        self.entities = []

    def upsert_entity(self, entity):
        self.entities.append(entity)

    def search(self, incoming, *, top_k=None):
        from app.EvoRAG.entity_store.models import CandidateEntity

        return [CandidateEntity(entity=self.entities[0], score=1.0, rank=1, source="fake")]


class FakeJudgeClient:
    def usage_checkpoint(self) -> int:
        return 0

    def usage_records_since(self, checkpoint: int = 0) -> list[dict]:
        return []

    async def chat_json(self, **kwargs):
        return {
            "decision": "matched",
            "matched_seed_id": "seed-001",
            "confidence": 0.95,
            "reason": "same entity",
        }

