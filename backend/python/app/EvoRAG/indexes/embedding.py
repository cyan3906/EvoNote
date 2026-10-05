import logging

from openai import AsyncOpenAI

from app.EvoRAG.config import EvoRAGSettings, settings

logger = logging.getLogger(__name__)


def _strip_quotes(value: str) -> str:
    return value.strip().strip('"\'')


class EvoRAGEmbeddingError(RuntimeError):
    pass


class EvoRAGEmbeddingClient:
    def __init__(self, config: EvoRAGSettings = settings) -> None:
        self.config = config
        self.api_key = _strip_quotes(config.embedding_api_key) or config.api_key
        self.api_base_url = _strip_quotes(config.embedding_api_base_url) or config.api_base_url

        # 诊断日志：确认实际使用的嵌入 API 配置
        source = "embedding专有" if _strip_quotes(config.embedding_api_base_url) else "主API回退"
        masked_key = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
        logger.info(
            "EvoRAGEmbeddingClient 初始化 | 来源=%s | base_url=%s | api_key=%s | model=%s",
            source,
            self.api_base_url or "OpenAI默认",
            masked_key,
            config.embedding_model,
        )

        self._client = AsyncOpenAI(
            api_key=self.api_key or "missing-evorag-api-key",
            base_url=self.api_base_url or None,
            timeout=config.timeout_seconds,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.api_key:
            raise EvoRAGEmbeddingError("EVORAG_EMBEDDING_API_KEY or EVORAG_API_KEY is not configured")
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