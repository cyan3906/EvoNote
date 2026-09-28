import asyncio
import os
from pathlib import Path

import pytest

from app.EvoRAG.entity_admission_eval import DEFAULT_OUTPUT_PATH, evaluate_entity_admission_suite


@pytest.mark.skipif(
    os.getenv("EVORAG_RUN_LIVE_EVAL") != "1",
    reason="Set EVORAG_RUN_LIVE_EVAL=1 to run the live LLM entity-admission evaluation.",
)
def test_live_entity_admission_eval_writes_report() -> None:
    limit_text = os.getenv("EVORAG_EVAL_LIMIT", "").strip()
    limit = int(limit_text) if limit_text else None
    output_path = Path(os.getenv("EVORAG_EVAL_OUTPUT", str(DEFAULT_OUTPUT_PATH)))

    report = asyncio.run(evaluate_entity_admission_suite(output_path=output_path, limit=limit))

    assert output_path.exists()
    expected_total = min(max(limit, 0), 80) if limit is not None else 80
    assert report["total_cases"] == expected_total
    assert "false_promotion_rate" in report["overall"]
    assert "unmatched_predicted_attribute_rate" in report["overall"]
