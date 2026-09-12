from pathlib import Path

from fastapi.testclient import TestClient

from app.core import database, evolution
from app.main import app


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post("/api/auth/login", json={"password": "evonote2026"})

    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def use_temp_database(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(database, "DATABASE_PATH", tmp_path / "evonote-test.sqlite3")


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
    client = TestClient(app)
    headers = auth_headers(client)

    first = client.post(
        "/api/notes",
        json={
            "title": "Redis RDB",
            "tags": "redis,rdb",
            "body": "Redis RDB 是快照持久化方式。RDB 会定期保存数据快照。",
        },
        headers=headers,
    )
    assert first.status_code == 201

    second = client.post(
        "/api/notes",
        json={
            "title": "RDB 复习",
            "tags": "redis,rdb",
            "body": "Redis RDB 是快照持久化方式。RDB 适合做备份。",
        },
        headers=headers,
    )
    assert second.status_code == 201

    state = client.post(f"/api/evolution/notes/{second.json()['id']}/scan", headers=headers)

    assert state.status_code == 200
    body = state.json()
    assert body["representation"]["l2_summary"]
    assert body["representation"]["keywords"]
    assert body["claims"]
    assert body["suggestions"]
    assert {suggestion["relation"] for suggestion in body["suggestions"]} & {"duplicate", "supplement"}


def test_merge_suggestion_can_be_applied(monkeypatch, tmp_path: Path) -> None:
    use_temp_database(monkeypatch, tmp_path)
    use_fake_evolution_model(monkeypatch)
    client = TestClient(app)
    headers = auth_headers(client)

    client.post(
        "/api/notes",
        json={
            "title": "Redis RDB",
            "tags": "redis,rdb",
            "body": "Redis RDB 是快照持久化方式。",
        },
        headers=headers,
    )
    source = client.post(
        "/api/notes",
        json={
            "title": "RDB 风险",
            "tags": "redis,rdb",
            "body": "Redis RDB 是快照持久化方式。RDB 可能丢失最近一次快照后的数据。",
        },
        headers=headers,
    )
    state = client.post(f"/api/evolution/notes/{source.json()['id']}/scan", headers=headers).json()
    suggestion_id = state["suggestions"][0]["id"]

    applied = client.post(f"/api/evolution/suggestions/{suggestion_id}/apply", headers=headers)

    assert applied.status_code == 200
    assert applied.json()["status"] == "applied"
