from pathlib import Path

from app.core import database, evolution, metrics


def use_temp_database(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "evonote-test.sqlite3")
    monkeypatch.setattr(metrics, "metrics_path", lambda: tmp_path / "metrics" / "evolution-runs.jsonl")
    monkeypatch.setattr(database, "_delete_external_note_indexes", lambda note_id: None)
    monkeypatch.setattr(evolution, "sync_note_indexes", lambda note, representation, claims: {})
    monkeypatch.setattr(evolution, "hybrid_retrieve_notes", lambda representation, exclude_note_id=None, top_k=None: [])
    monkeypatch.setattr(
        evolution,
        "embed_texts",
        lambda texts, fallback_vector=None: (
            [(fallback_vector or evolution.text_vector)(text) for text in texts],
            None,
        ),
    )


def use_fake_evolution_model(monkeypatch) -> None:
    def fake_analysis(note: dict[str, str], raw_blocks: list[dict[str, str]]) -> dict[str, object]:
        claim_text = "Redis RDB 是快照持久化方式"

        if "备份" in note["body"]:
            claim_text = "RDB 适合做备份"
        elif "丢失" in note["body"]:
            claim_text = "RDB 可能丢失最近一次快照后的数据"

        return {
            "note": {
                "l2_summary": f"{note['title']} 的摘要",
                "l3_text": "Redis RDB 快照 持久化 备份",
                "keywords": ["Redis", "RDB", "快照", "持久化", "备份"],
            },
            "blocks": [
                {
                    "block_index": index,
                    "l2_summary": block["text"][:80] or "空白内容",
                    "l3_text": "Redis RDB 快照 持久化 备份",
                    "keywords": ["Redis", "RDB", "快照", "持久化", "备份"],
                    "claims": [
                        {
                            "claim_text": claim_text,
                            "subject": "Redis RDB",
                            "predicate": "是" if "是" in claim_text else "适合",
                            "object_text": claim_text,
                            "source_text": claim_text,
                            "keywords": ["Redis", "RDB", "快照", "持久化", "备份"],
                            "confidence": 0.92,
                        }
                    ],
                }
                for index, block in enumerate(raw_blocks)
            ],
        }

    monkeypatch.setattr(evolution, "analyze_note_with_model", fake_analysis)


def test_note_scan_creates_l2_l3_claims_and_suggestions(monkeypatch, tmp_path: Path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    use_fake_evolution_model(monkeypatch)

    first = database.create_note(
        title="Redis RDB",
        tags="redis,rdb",
        body="Redis RDB 是快照持久化方式。RDB 会定期保存数据快照。",
    )
    second = database.create_note(
        title="RDB 复习",
        tags="redis,rdb",
        body="Redis RDB 是快照持久化方式。RDB 适合做备份。",
    )

    first_state = evolution.scan_note(first["id"])
    assert first_state is not None

    state = evolution.scan_note(second["id"])

    assert state is not None
    body = state
    assert body["representation"]["l2_summary"]
    assert body["representation"]["keywords"]
    assert body["claims"]
    assert body["suggestions"]
    assert {suggestion["relation"] for suggestion in body["suggestions"]} & {"duplicate", "supplement"}


def test_merge_suggestion_can_be_applied(monkeypatch, tmp_path: Path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    use_fake_evolution_model(monkeypatch)

    first = database.create_note(
        title="Redis RDB",
        tags="redis,rdb",
        body="Redis RDB 是快照持久化方式。",
    )
    source = database.create_note(
        title="RDB 风险",
        tags="redis,rdb",
        body="Redis RDB 是快照持久化方式。RDB 可能丢失最近一次快照后的数据。",
    )
    first_state = evolution.scan_note(first["id"])
    assert first_state is not None

    state = evolution.scan_note(source["id"])
    assert state is not None
    suggestion_id = state["suggestions"][0]["id"]

    applied = evolution.apply_suggestion(suggestion_id)

    assert applied is not None
    assert applied["status"] == "applied"
