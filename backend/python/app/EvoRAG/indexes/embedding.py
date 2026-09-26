from openai import AsyncOpenAI

from app.EvoRAG.config import EvoRAGSettings, settings


class EvoRAGEmbeddingError(RuntimeError):
    pass


class EvoRAGEmbeddingClient:
    def __init__(self, config: EvoRAGSettings = settings) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key or "missing-evorag-api-key",
            base_url=config.api_base_url or None,
            timeout=config.timeout_seconds,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.config.api_key:
            raise EvoRAGEmbeddingError("EVORAG_API_KEY is not configured")
        if not self.config.embedding_model:
            raise EvoRAGEmbeddingError("EVORAG_EMBEDDING_MODEL is not configured")

        normalized = [str(text or "").strip() for text in texts]
        if not normalized:
            return []

        try:
            response = await self._client.embeddings.create(
                model=self.config.embedding_model,
                input=normalized,
                dimensions=self.config.embedding_dimensions,
            )
        except Exception:
            response = await self._client.embeddings.create(
                model=self.config.embedding_model,
                input=normalized,
            )

        vectors = [list(item.embedding) for item in response.data]
        return [normalize_vector(vector) for vector in vectors]

    async def embed_text(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0] if vectors else []


def normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm <= 0:
        return vector
    return [value / norm for value in vector]
