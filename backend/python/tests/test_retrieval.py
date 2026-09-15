from app.core import retrieval
from app.core.retrieval import RetrievalHit, build_claim_documents, reciprocal_rank_fusion


def test_reciprocal_rank_fusion_combines_vector_and_keyword_rankings() -> None:
    vector_hits = [
        RetrievalHit(claim_id="A", note_id="N1", score=0.9, rank=1, source="milvus"),
        RetrievalHit(claim_id="B", note_id="N2", score=0.8, rank=2, source="milvus"),
    ]
    keyword_hits = [
        RetrievalHit(claim_id="B", note_id="N2", score=12.0, rank=1, source="elasticsearch"),
        RetrievalHit(claim_id="C", note_id="N3", score=8.0, rank=2, source="elasticsearch"),
    ]

    fused = reciprocal_rank_fusion([vector_hits, keyword_hits], rrf_k=60, top_k=3)

    assert [hit.claim_id for hit in fused] == ["B", "A", "C"]
    assert fused[0].source == "milvus+elasticsearch"


def test_build_claim_documents_prefers_claim_vector(monkeypatch) -> None:
    monkeypatch.setattr(retrieval.settings, "embedding_dimensions", 3)
    note = {"id": "note-1", "title": "Redis", "tags": "redis"}
    representation = {
        "l2_summary": "Redis persistence",
        "l3_text": "Redis RDB AOF",
        "keywords": ["Redis"],
        "vector": [9.0, 9.0, 9.0],
    }
    claims = [
        {
            "id": "claim-1",
            "note_id": "note-1",
            "claim_text": "RDB is a snapshot persistence mode",
            "keywords": ["RDB"],
            "vector": [1.0, 2.0, 3.0],
        }
    ]

    documents = build_claim_documents(note, representation, claims)

    assert documents[0]["vector"] == [1.0, 2.0, 3.0]


def test_build_claim_documents_does_not_fall_back_to_note_vector(monkeypatch) -> None:
    monkeypatch.setattr(retrieval.settings, "embedding_dimensions", 3)
    monkeypatch.setattr(retrieval, "safe_embed_text", lambda text: [])
    note = {"id": "note-1", "title": "Redis", "tags": "redis"}
    representation = {
        "l2_summary": "Redis persistence",
        "l3_text": "Redis RDB AOF",
        "keywords": ["Redis"],
        "vector": [9.0, 9.0, 9.0],
    }
    claims = [
        {
            "id": "claim-1",
            "note_id": "note-1",
            "claim_text": "RDB is a snapshot persistence mode",
            "keywords": ["RDB"],
            "vector": [],
        }
    ]

    documents = build_claim_documents(note, representation, claims)

    assert documents[0]["vector"] == []
