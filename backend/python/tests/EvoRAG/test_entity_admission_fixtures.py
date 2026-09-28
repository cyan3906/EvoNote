import json
from pathlib import Path


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "entity_admission"

EXPECTED_FILES = {
    "entity_vs_attribute.json": "Entity vs Attribute",
    "entity_vs_process_step.json": "Entity vs Process Step",
    "entity_vs_section_title.json": "Entity vs Section Title",
    "context_independence.json": "Context Independence",
    "granularity.json": "Granularity",
    "alias_coreference.json": "Alias / Coreference",
    "ambiguous_entity.json": "Ambiguous Entity",
    "cross_block_entity.json": "Cross-block Entity",
}

ATTRIBUTE_TYPES = {
    "definition",
    "purpose",
    "core_idea",
    "mechanism",
    "components",
    "constraints",
    "related",
}

NON_ENTITY_ROLES = {
    "attribute",
    "step",
    "section_topic",
    "condition",
    "mention",
    "alias",
    "component",
}


def test_entity_admission_fixture_suite_shape() -> None:
    total_cases = 0
    for filename, category in EXPECTED_FILES.items():
        path = FIXTURE_DIR / filename
        assert path.exists(), f"missing fixture file: {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["category"] == category
        assert payload["case_count"] == 10
        cases = payload["cases"]
        assert len(cases) == 10
        total_cases += len(cases)

        for case in cases:
            assert case["case_id"].startswith(filename.removesuffix(".json"))
            assert case["category"] == category
            assert 200 <= len(case["input_text"]) <= 400
            assert case["expected_entities"], case["case_id"]
            assert case["expected_non_entities"], case["case_id"]

            for entity in case["expected_entities"]:
                assert entity["name"]
                assert entity["entity_type"]
                attributes = entity.get("attributes", {})
                assert attributes
                assert set(attributes).issubset(ATTRIBUTE_TYPES)
                assert any(attributes.values()), entity["name"]

            for item in case["expected_non_entities"]:
                assert item["text"]
                assert item["expected_role"] in NON_ENTITY_ROLES
                assert item["parent_entity"]
                assert item["reason"]

    assert total_cases == 80
