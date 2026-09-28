from app.EvoRAG.entity_admission_eval import compute_metrics, summarize_llm_usage


def test_compute_entity_admission_eval_metrics_from_judgments() -> None:
    judgments = [
        {
            "case_id": "case_001",
            "gold_entity_matches": [
                {"gold_entity": "Redis 分布式锁", "status": "matched"},
                {"gold_entity": "Lua Script", "status": "missing"},
            ],
            "predicted_entity_judgments": [
                {"predicted_entity": "Redis 锁", "status": "correct"},
                {"predicted_entity": "自动续期", "status": "incorrect"},
                {"predicted_entity": "Lua Script", "status": "correct"},
            ],
            "non_entity_judgments": [
                {"text": "自动续期", "promoted": True},
                {"text": "释放锁", "promoted": False},
                {"text": "m_ids", "promoted": False},
                {"text": "RR", "promoted": False},
            ],
            "predicted_attribute_judgments": [
                {"predicted_attribute": "SET NX EX 获取锁", "status": "matched", "bucket_correct": True},
                {"predicted_attribute": "后台线程续期", "status": "matched", "bucket_correct": False},
                {"predicted_attribute": "Redlock 保证安全", "status": "unmatched", "bucket_correct": False},
                {"predicted_attribute": "直接 DEL 释放", "status": "contradictory", "bucket_correct": False},
            ],
            "gold_attribute_matches": [
                {"gold_attribute": "SET NX EX 获取锁", "status": "matched"},
                {"gold_attribute": "释放前比较 token", "status": "missing"},
            ],
        }
    ]

    metrics = compute_metrics(judgments)

    assert metrics == {
        "entity_precision": 0.666667,
        "entity_recall": 0.5,
        "false_promotion_rate": 0.25,
        "matched_predicted_attribute_rate": 0.5,
        "unmatched_predicted_attribute_rate": 0.25,
        "contradictory_predicted_attribute_rate": 0.25,
        "wrong_bucket_rate": 0.5,
        "gold_attribute_recall": 0.5,
        "counts": {
            "gold_entities": 2,
            "matched_gold_entities": 1,
            "predicted_entities": 3,
            "correct_predicted_entities": 2,
            "gold_non_entities": 4,
            "false_promotions": 1,
            "predicted_attributes": 4,
            "matched_predicted_attributes": 2,
            "unmatched_predicted_attributes": 1,
            "contradictory_predicted_attributes": 1,
            "bucket_checked_predicted_attributes": 2,
            "wrong_bucket_predicted_attributes": 1,
            "gold_attributes": 2,
            "matched_gold_attributes": 1,
        },
    }


def test_summarize_llm_usage_records_tokens_and_duration() -> None:
    records = [
        {
            "operation_name": "EvoRAG block split",
            "model": "reasoner",
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "duration_ms": 120.125,
        },
        {
            "operation_name": "EvoRAG entity extraction block 0",
            "model": "reasoner",
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "total_tokens": 100,
            "duration_ms": 70.335,
        },
        {
            "operation_name": "EvoRAG eval judge case_001",
            "model": "deepseek-v4-flash-0731",
            "prompt_tokens": 60,
            "completion_tokens": 40,
            "total_tokens": 100,
            "duration_ms": 50.0,
        },
    ]

    summary = summarize_llm_usage(records)

    assert summary == {
        "calls": 3,
        "prompt_tokens": 240,
        "completion_tokens": 85,
        "total_tokens": 325,
        "total_duration_ms": 240.46,
        "avg_duration_ms": 80.153333,
        "by_model": {
            "reasoner": {
                "calls": 2,
                "prompt_tokens": 180,
                "completion_tokens": 45,
                "total_tokens": 225,
                "total_duration_ms": 190.46,
                "avg_duration_ms": 95.23,
            },
            "deepseek-v4-flash-0731": {
                "calls": 1,
                "prompt_tokens": 60,
                "completion_tokens": 40,
                "total_tokens": 100,
                "total_duration_ms": 50.0,
                "avg_duration_ms": 50.0,
            },
        },
    }
