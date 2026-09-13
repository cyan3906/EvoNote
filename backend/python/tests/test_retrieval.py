from app.core.retrieval import RetrievalHit, reciprocal_rank_fusion


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
