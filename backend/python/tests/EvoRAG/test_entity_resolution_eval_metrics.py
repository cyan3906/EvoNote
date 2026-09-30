import asyncio
import json
from pathlib import Path

from app.EvoRAG.entity_resolution_eval import (
    MEMORY_GUARDED_HYBRID_STRATEGY,
    build_parameterized_output_path,
    compute_flow_metrics,
    compute_deepseek_decision_metrics,
    compute_retrieval_metrics,
    evaluate_entity_resolution_suite,
    get_entity_resolution_strategy,
    load_benchmark,
    preflight_index_connections,
)
from app.EvoRAG.entity_store.models import EntityRelationMemoryRecord, EntityScope, StoredEntity


def test_load_benchmark_preserves_unified_scope_and_counts() -> None:
    benchmark = load_benchmark()

    assert benchmark["dataset_id"] == "entity-resolution-benchmark-v1"
    assert benchmark["scope"]["collection_id"] == "benchmark-v1"
    assert len(benchmark["seed_entities"]) == 70
    assert len(benchmark["cases"]) == 60


def test_build_parameterized_output_path_separates_runs_by_eval_parameters() -> None:
    output_path = build_parameterized_output_path(
        strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
        category="nil_new_entity",
        limit=10,
        top_k=5,
        rebuild_seeds=True,
    )

    assert output_path.parent.name == "memory_guarded_hybrid__category-nil_new_entity__limit-10__top-k-5__rebuild-seeds"
    assert output_path.name.startswith("entity_resolution_eval_report__")
    assert output_path.suffix == ".json"


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


def test_metrics_use_none_for_not_applicable_rates_without_denominator() -> None:
    case_reports = [
        {
            "category": "nil_new_entity",
            "expected": {"decision": "new", "matched_seed_id": None},
            "judgment": {"decision": "new", "matched_seed_id": None},
            "retrieval": {"ranked_seed_ids": ["seed-001"]},
        }
    ]

    retrieval_metrics = compute_retrieval_metrics(case_reports)
    judge_metrics = compute_deepseek_decision_metrics(case_reports)

    assert retrieval_metrics["recall@1"] is None
    assert retrieval_metrics["recall@3"] is None
    assert retrieval_metrics["recall@5"] is None
    assert retrieval_metrics["mrr"] is None
    assert retrieval_metrics["ambiguous_coverage@3"] is None
    assert judge_metrics["matched_entity_accuracy"] is None
    assert judge_metrics["false_new_rate"] is None
    assert judge_metrics["ambiguous_accuracy"] is None


def test_compute_flow_metrics_groups_stage_by_category_and_accuracy() -> None:
    case_reports = [
        {
            "category": "nil_new_entity",
            "expected": {"decision": "new", "matched_seed_id": None},
            "judgment": {"decision": "new", "matched_seed_id": None},
            "flow": {
                "stage": "llm_admission_guard_reject",
                "used_hybrid": False,
                "wrote_experience": False,
                "manual_review_triggered": False,
            },
        },
        {
            "category": "alias_match",
            "expected": {"decision": "matched", "matched_seed_id": "seed-001"},
            "judgment": {"decision": "matched", "matched_seed_id": "seed-001"},
            "flow": {
                "stage": "mysql_relation_memory_judge",
                "used_hybrid": False,
                "wrote_experience": False,
                "manual_review_triggered": False,
            },
        },
        {
            "category": "related_but_different",
            "expected": {"decision": "new", "matched_seed_id": None},
            "judgment": {"decision": "matched", "matched_seed_id": "seed-002"},
            "flow": {
                "stage": "memory_judge_fallback_hybrid",
                "used_hybrid": True,
                "wrote_experience": True,
                "manual_review_triggered": True,
            },
        },
    ]

    metrics = compute_flow_metrics(case_reports)

    assert metrics["by_stage"] == {
        "llm_admission_guard_reject": 1,
        "mysql_relation_memory_judge": 1,
        "memory_judge_fallback_hybrid": 1,
    }
    assert metrics["by_category_stage"] == {
        "alias_match": {"mysql_relation_memory_judge": 1},
        "nil_new_entity": {"llm_admission_guard_reject": 1},
        "related_but_different": {"memory_judge_fallback_hybrid": 1},
    }
    assert metrics["accuracy_by_stage"]["memory_judge_fallback_hybrid"]["accuracy"] == 0.0
    assert metrics["accuracy_by_stage"]["mysql_relation_memory_judge"]["accuracy"] == 1.0
    assert metrics["hybrid_fallback_rate"] == 0.333333
    assert metrics["experience_write_count"] == 1
    assert metrics["manual_review_trigger_count"] == 1


def test_evaluate_entity_resolution_suite_uses_fakes_and_emits_progress(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    progress: list[str] = []
    hybrid_index = FakeHybridIndex(seeds_ready=False)

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(),
            progress=progress.append,
            strategy="hybrid_rrf_baseline",
        )
    )

    assert output_path.exists()
    assert report["strategy"] == "hybrid_rrf_baseline"
    assert any("加载数据集 tiny" in item for item in progress)
    assert any("写入 seed 实体 1/1: seed-001 MVCC" in item for item in progress)
    assert any("处理 case 1/1: case-001 alias_match" in item for item in progress)
    assert hybrid_index.upsert_count == 1
    assert report["total_cases"] == 1
    assert report["overall"]["retrieval"]["recall@1"] == 1.0
    assert report["overall"]["deepseek_judge"]["decision_accuracy"] == 1.0
    assert report["cases"][0]["flow"]["stage"] == "hybrid_rrf"
    assert report["cases"][0]["flow"]["used_hybrid"] is True
    assert report["flow_metrics"]["by_stage"] == {"hybrid_rrf": 1}


def test_get_entity_resolution_strategy_rejects_unknown_strategy() -> None:
    try:
        get_entity_resolution_strategy("unknown")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected unknown strategy to fail")

    assert "Unsupported entity resolution strategy" in message
    assert "hybrid_rrf_baseline" in message


def test_get_entity_resolution_strategy_accepts_memory_guarded_hybrid() -> None:
    strategy = get_entity_resolution_strategy(MEMORY_GUARDED_HYBRID_STRATEGY)

    assert strategy is not None


def test_evaluate_entity_resolution_suite_skips_existing_seed_dataset(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    progress: list[str] = []
    embedding_client = FakeEmbeddingClient()
    hybrid_index = FakeHybridIndex(seeds_ready=True)

    asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=embedding_client,
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(),
            progress=progress.append,
        )
    )

    assert any("评测 seed 已存在，跳过写入" in item for item in progress)
    assert not any("写入 seed 实体" in item for item in progress)
    assert embedding_client.seed_embed_batches == []
    assert hybrid_index.upsert_count == 0


def test_evaluate_entity_resolution_suite_rebuilds_seed_dataset_when_requested(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    progress: list[str] = []
    embedding_client = FakeEmbeddingClient()
    hybrid_index = FakeHybridIndex(seeds_ready=True)

    asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=embedding_client,
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(),
            progress=progress.append,
            rebuild_seeds=True,
        )
    )

    assert any("收到 --rebuild-seeds" in item for item in progress)
    assert any("写入 seed 实体 1/1: seed-001 MVCC" in item for item in progress)
    assert len(embedding_client.seed_embed_batches) == 1
    assert hybrid_index.upsert_count == 1


def test_memory_guarded_hybrid_uses_direct_allow_without_hybrid_search(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True)
    relation_memory = FakeRelationMemory(direct_relation=memory_record(decision="allow"))

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(),
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert hybrid_index.search_count == 0
    assert report["cases"][0]["judgment"]["decision"] == "matched"
    assert report["cases"][0]["judgment"]["matched_seed_id"] == "seed-001"
    assert report["cases"][0]["strategy_trace"]["stage"] == "redis_direct_allow"


def test_memory_guarded_hybrid_blocks_non_entity_after_redis_before_mysql(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True)
    relation_memory = FakeRelationMemory()
    judge_client = FakeJudgeClient(
        admission_response={
            "decision": "attribute",
            "confidence": 0.9,
            "reason": "incoming is an attribute, not an entity",
        }
    )

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=judge_client,
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert hybrid_index.search_count == 0
    assert relation_memory.list_count == 0
    assert len(judge_client.calls) == 1
    assert report["cases"][0]["judgment"] == {
        "decision": "new",
        "matched_seed_id": None,
        "confidence": 0.9,
        "reason": "incoming is an attribute, not an entity",
    }
    assert report["cases"][0]["strategy_trace"] == {
        "strategy": MEMORY_GUARDED_HYBRID_STRATEGY,
        "stage": "llm_admission_guard_reject",
        "admission_decision": "attribute",
    }

def test_memory_guarded_hybrid_uses_relation_judge_match_without_hybrid_search(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True)
    relation_memory = FakeRelationMemory(top_relations=[memory_record(decision="allow")])
    judge_client = FakeJudgeClient(
        memory_response={
            "decision": "matched",
            "matched_entity_id": 1,
            "confidence": 0.91,
            "reason": "white list confirms same entity",
        }
    )

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=judge_client,
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert hybrid_index.search_count == 0
    assert len(judge_client.calls) == 2
    assert report["cases"][0]["strategy_trace"]["stage"] == "mysql_relation_memory_judge"
    assert report["cases"][0]["judgment"]["confidence"] == 0.91


def test_memory_guarded_hybrid_blocks_attribute_relation_after_redis(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True)
    relation_memory = FakeRelationMemory(top_relations=[memory_record(decision="reject")])
    judge_client = FakeJudgeClient(
        memory_response={
            "decision": "attribute_of_entity",
            "matched_entity_id": 1,
            "confidence": 0.89,
            "reason": "incoming is an attribute of the candidate entity",
        }
    )

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=judge_client,
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert hybrid_index.search_count == 0
    assert len(judge_client.calls) == 2
    assert report["cases"][0]["judgment"] == {
        "decision": "new",
        "matched_seed_id": None,
        "confidence": 0.89,
        "reason": "incoming is an attribute of the candidate entity",
    }
    assert report["cases"][0]["strategy_trace"]["stage"] == "llm_relation_guard_reject"
    assert report["cases"][0]["strategy_trace"]["relation_guard_decision"] == "attribute_of_entity"
    assert relation_memory.recorded_relations == [
        {
            "candidate_id": 1,
            "decision": "reject",
            "relation_type": "attribute_of_entity",
            "confidence": 0.89,
            "source": "llm_guard",
        }
    ]

def test_memory_guarded_hybrid_falls_back_to_hybrid_when_memory_rejects(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True)
    relation_memory = FakeRelationMemory(top_relations=[memory_record(decision="reject")])
    judge_client = FakeJudgeClient(
        memory_response={
            "decision": "not_matched",
            "matched_entity_id": None,
            "confidence": 0.82,
            "reason": "black list says related but different",
        }
    )

    report = asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=judge_client,
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert hybrid_index.search_count == 1
    assert len(judge_client.calls) == 3
    assert report["cases"][0]["strategy_trace"]["stage"] == "memory_judge_fallback_hybrid"


def test_memory_guarded_hybrid_records_low_score_direct_reject_experience(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True, search_score=0.49)
    relation_memory = FakeRelationMemory()

    asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(),
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert relation_memory.recorded_relations == [
        {
            "candidate_id": 1,
            "decision": "reject",
            "relation_type": "low_score_direct_reject",
            "confidence": 0.51,
            "source": "auto",
        }
    ]


def test_memory_guarded_hybrid_records_manual_band_matched_experience(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True, search_score=0.8)
    relation_memory = FakeRelationMemory()

    asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(),
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert relation_memory.recorded_relations == [
        {
            "candidate_id": 1,
            "decision": "allow",
            "relation_type": "manual_match",
            "confidence": 1.0,
            "source": "manual",
        }
    ]


def test_memory_guarded_hybrid_records_manual_band_rejected_experience(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True, search_score=0.8)
    relation_memory = FakeRelationMemory()

    asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(
                baseline_response={
                    "decision": "new",
                    "matched_seed_id": None,
                    "confidence": 0.73,
                    "reason": "manual decision says different",
                }
            ),
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert relation_memory.recorded_relations == [
        {
            "candidate_id": 1,
            "decision": "reject",
            "relation_type": "manual_reject",
            "confidence": 1.0,
            "source": "manual",
        }
    ]


def test_memory_guarded_hybrid_records_high_score_llm_reject_when_final_judge_says_new(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(seeds_ready=True, search_score=0.95)
    relation_memory = FakeRelationMemory()

    asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(
                baseline_response={
                    "decision": "new",
                    "matched_seed_id": None,
                    "confidence": 0.96,
                    "reason": "top candidate is related but not the same entity",
                }
            ),
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert relation_memory.recorded_relations == [
        {
            "candidate_id": 1,
            "decision": "reject",
            "relation_type": "high_score_llm_reject",
            "confidence": 1.0,
            "source": "llm_judge",
        }
    ]


def test_memory_guarded_hybrid_records_manual_match_when_es_milvus_scores_diverge(tmp_path: Path) -> None:
    fixture_path = write_tiny_resolution_fixture(tmp_path)
    output_path = tmp_path / "report.json"
    hybrid_index = FakeHybridIndex(
        seeds_ready=True,
        search_score=0.95,
        vector_score=0.95,
        es_score=0.42,
    )
    relation_memory = FakeRelationMemory()

    asyncio.run(
        evaluate_entity_resolution_suite(
            fixture_path=fixture_path,
            output_path=output_path,
            embedding_client=FakeEmbeddingClient(),
            hybrid_index=hybrid_index,
            judge_client=FakeJudgeClient(),
            strategy=MEMORY_GUARDED_HYBRID_STRATEGY,
            relation_memory=relation_memory,
        )
    )

    assert relation_memory.recorded_relations == [
        {
            "candidate_id": 1,
            "decision": "allow",
            "relation_type": "manual_match",
            "confidence": 1.0,
            "source": "manual",
        }
    ]


def test_preflight_index_connections_retries_with_backoff_until_success() -> None:
    progress: list[str] = []
    sleeps: list[float] = []
    index = FlakyPreflightIndex(failures_before_success=2)

    preflight_index_connections(index, progress=progress.append, sleep_fn=sleeps.append)

    assert index.attempts == 3
    assert sleeps == [2, 4]
    assert any("连接评测索引失败" in item and "2 秒后重试" in item for item in progress)
    assert any("连接评测索引成功" in item for item in progress)


def test_preflight_index_connections_reports_failure_after_retries() -> None:
    progress: list[str] = []
    sleeps: list[float] = []
    index = FlakyPreflightIndex(failures_before_success=10)

    try:
        preflight_index_connections(index, progress=progress.append, sleep_fn=sleeps.append)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected preflight_index_connections to fail after retries")

    assert index.attempts == 5
    assert sleeps == [2, 4, 8, 16]
    assert "评测索引连接失败" in message
    assert any("连接评测索引最终失败" in item for item in progress)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.seed_embed_batches: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.seed_embed_batches.append(texts)
        return [[1.0, 0.0] for _ in texts]

    async def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeHybridIndex:
    def __init__(
        self,
        *,
        seeds_ready: bool = False,
        search_score: float = 1.0,
        vector_score: float = 0.0,
        es_score: float = 0.0,
    ) -> None:
        self.seeds_ready = seeds_ready
        self.search_score = search_score
        self.vector_score = vector_score
        self.es_score = es_score
        self.entities = []
        self.upsert_count = 0
        self.search_count = 0

    def ensure_indexes(self):
        return None

    def seed_entities_exist(self, entities):
        return self.seeds_ready

    def upsert_entity(self, entity):
        self.entities.append(entity)
        self.upsert_count += 1

    def search(self, incoming, *, top_k=None):
        from app.EvoRAG.entity_store.models import CandidateEntity
        from app.EvoRAG.entity_store.models import EntityScope, StoredEntity

        self.search_count += 1
        entity = self.entities[0] if self.entities else StoredEntity(
            id=1,
            canonical_name="MVCC",
            normalized_name="mvcc",
            entity_type="concept",
            scope=EntityScope(workspace_id="eval", project_id="entity-resolution", collection_id="tiny", domain="computer-science"),
        )
        return [
            CandidateEntity(
                entity=entity,
                score=self.search_score,
                rank=1,
                source="fake",
                vector_score=self.vector_score,
                es_score=self.es_score,
            )
        ]


class FakeJudgeClient:
    def __init__(
        self,
        *,
        admission_response: dict | None = None,
        memory_response: dict | None = None,
        baseline_response: dict | None = None,
    ) -> None:
        self.admission_response = admission_response or {
            "decision": "entity",
            "confidence": 0.96,
            "reason": "valid entity candidate",
        }
        self.memory_response = memory_response
        self.baseline_response = baseline_response or {
            "decision": "matched",
            "matched_seed_id": "seed-001",
            "confidence": 0.95,
            "reason": "same entity",
        }
        self.calls: list[dict] = []

    def usage_checkpoint(self) -> int:
        return len(self.calls)

    def usage_records_since(self, checkpoint: int = 0) -> list[dict]:
        return []

    async def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        operation_name = str(kwargs.get("operation_name") or "")
        if "admission guard" in operation_name:
            return self.admission_response
        if "relation memory" in operation_name and self.memory_response is not None:
            return self.memory_response
        return self.baseline_response


class FakeRelationMemory:
    def __init__(
        self,
        *,
        direct_relation: EntityRelationMemoryRecord | None = None,
        top_relations: list[EntityRelationMemoryRecord] | None = None,
    ) -> None:
        self.direct_relation = direct_relation
        self.top_relations = top_relations or []
        self.recorded_relations: list[dict] = []
        self.list_count = 0

    def get_direct_relation(self, incoming):
        return self.direct_relation

    def list_top_relations(self, incoming, *, limit: int = 30):
        self.list_count += 1
        return self.top_relations[:limit]

    def record_relation(self, *, incoming, candidate, decision, relation_type, confidence, source, reason):
        self.recorded_relations.append(
            {
                "candidate_id": candidate.id,
                "decision": decision,
                "relation_type": relation_type,
                "confidence": confidence,
                "source": source,
            }
        )


class FlakyPreflightIndex:
    def __init__(self, *, failures_before_success: int) -> None:
        self.failures_before_success = failures_before_success
        self.attempts = 0

    def ensure_indexes(self) -> None:
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise TimeoutError(f"timeout {self.attempts}")


def write_tiny_resolution_fixture(tmp_path: Path) -> Path:
    fixture_path = tmp_path / "benchmark.json"
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
    return fixture_path


def memory_record(*, decision: str) -> EntityRelationMemoryRecord:
    scope = EntityScope(workspace_id="eval", project_id="entity-resolution", collection_id="tiny", domain="computer-science")
    return EntityRelationMemoryRecord(
        id=101,
        decision=decision,
        relation_type="same_entity" if decision == "allow" else "related_but_different",
        candidate=StoredEntity(
            id=1,
            canonical_name="MVCC",
            normalized_name="mvcc",
            entity_type="concept",
            scope=scope,
            aliases=["多版本并发控制"],
            identity_description="数据库多版本并发控制机制。",
        ),
        confidence=0.88,
        hit_count=7,
        source="test",
        reason="fixture relation",
    )







