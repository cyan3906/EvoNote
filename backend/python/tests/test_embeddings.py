from __future__ import annotations

from app.core import embeddings


class FakeEmbedding:
    def __init__(self, values: list[float]) -> None:
        self.embedding = values


class FakeEmbeddingResponse:
    def __init__(self, texts: list[str]) -> None:
        self.data = [
            FakeEmbedding([float(len(text)), float(index)])
            for index, text in enumerate(texts)
        ]


class FakeEmbeddingsClient:
    def __init__(self, calls: list[list[str]]) -> None:
        self.calls = calls

    def create(self, *, model: str, input: list[str]) -> FakeEmbeddingResponse:
        self.calls.append(list(input))
        return FakeEmbeddingResponse(input)


class FakeOpenAI:
    calls: list[list[str]] = []

    def __init__(self, **kwargs: object) -> None:
        self.embeddings = FakeEmbeddingsClient(self.calls)


def reset_breaker() -> None:
    embeddings._BREAKER = embeddings.CircuitBreaker(failure_threshold=5, cooldown_seconds=30)


def test_embed_texts_batches_and_preserves_order(monkeypatch) -> None:
    reset_breaker()
    FakeOpenAI.calls = []
    monkeypatch.setattr(embeddings, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(embeddings.settings, "embedding_api_key", "test-key")
    monkeypatch.setattr(embeddings.settings, "agent_api_key", "")
    monkeypatch.setattr(embeddings.settings, "embedding_model", "test-embedding")

    vectors, stats = embeddings.embed_texts(
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        batch_size=2,
        max_concurrency=1,
    )

    assert FakeOpenAI.calls == [["alpha", "beta"], ["gamma", "delta"], ["epsilon"]]
    assert vectors == [[5.0, 0.0], [4.0, 1.0], [5.0, 0.0], [5.0, 1.0], [7.0, 0.0]]
    assert stats.batches == 3
    assert stats.embedded == 5
    assert stats.fallback == 0
    assert stats.failed_batches == 0


def test_embed_texts_deduplicates_and_fans_out_results(monkeypatch) -> None:
    reset_breaker()
    FakeOpenAI.calls = []
    monkeypatch.setattr(embeddings, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(embeddings.settings, "embedding_api_key", "test-key")
    monkeypatch.setattr(embeddings.settings, "agent_api_key", "")

    vectors, stats = embeddings.embed_texts(
        ["same", "other", "same"],
        batch_size=32,
        max_concurrency=4,
    )

    assert FakeOpenAI.calls == [["same", "other"]]
    assert vectors[0] == vectors[2]
    assert vectors[1] != vectors[0]
    assert stats.batches == 1
    assert stats.embedded == 3


def test_embed_texts_uses_fallback_without_api_key(monkeypatch) -> None:
    reset_breaker()
    monkeypatch.setattr(embeddings.settings, "embedding_api_key", "")
    monkeypatch.setattr(embeddings.settings, "agent_api_key", "")

    vectors, stats = embeddings.embed_texts(
        ["alpha", "beta"],
        fallback_vector=lambda text: [float(len(text))],
    )

    assert vectors == [[5.0], [4.0]]
    assert stats.embedded == 0
    assert stats.fallback == 2
