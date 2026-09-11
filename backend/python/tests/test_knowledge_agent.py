from fastapi.testclient import TestClient

from app.agents.knowledge_agent import (
    DEFAULT_KNOWLEDGE_ITEMS,
    DEFAULT_SOURCE_KNOWLEDGE,
    KnowledgeAssociationAgent,
)
from app.main import app


def test_agent_compares_one_source_with_default_candidates() -> None:
    result = KnowledgeAssociationAgent().associate(source=DEFAULT_SOURCE_KNOWLEDGE)

    assert result.source == DEFAULT_SOURCE_KNOWLEDGE
    assert result.candidates == DEFAULT_KNOWLEDGE_ITEMS
    assert len(result.candidates) == 5


def test_agent_captures_allowed_relation_types() -> None:
    result = KnowledgeAssociationAgent().associate(source=DEFAULT_SOURCE_KNOWLEDGE)
    relation_types = {relation.relation for relation in result.relations}

    assert relation_types
    assert relation_types.issubset({"包括", "关联", "冲突"})


def test_knowledge_graph_api() -> None:
    client = TestClient(app)

    response = client.get("/api/knowledge/graph")

    assert response.status_code == 200
    body = response.json()
    assert body["source"]["id"] == DEFAULT_SOURCE_KNOWLEDGE.id
    assert len(body["candidates"]) == 5
    assert "merged_content" in body


def test_associate_knowledge_api_uses_request_source() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/knowledge/associate",
        json={
            "source": {
                "id": "REQ-1",
                "title": "HTTP 与 TCP 的关系",
                "content": "HTTP 是应用层协议，通常基于 TCP 连接传输请求和响应。",
                "category": "计算机网络",
                "keywords": ["HTTP", "TCP", "连接"],
                "tags": ["网络"],
            }
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source"]["id"] == "REQ-1"
    assert len(body["candidates"]) == 5