import asyncio
from typing import Any

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.llm import EvoRAGLLMClient
from app.EvoRAG.models import EvoRAGPreprocessResult
from app.EvoRAG.services.block_splitter import BlockSplitter
from app.EvoRAG.services.entity_extractor import EntityExtractor


class EvoRAGProcessorError(RuntimeError):
    pass


class EvoRAGProcessor:
    def __init__(
        self,
        config: EvoRAGSettings = settings,
        llm_client: EvoRAGLLMClient | None = None,
        block_splitter: BlockSplitter | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self.config = config
        self.llm = llm_client or EvoRAGLLMClient(config)
        self.block_splitter = block_splitter or BlockSplitter(self.llm, config)
        self.entity_extractor = entity_extractor or EntityExtractor(self.llm)

    async def preprocess(self, text: str) -> EvoRAGPreprocessResult:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise EvoRAGProcessorError("input text is empty")

        try:
            blocks = await self.block_splitter.split(normalized_text)
            extractions = await self.entity_extractor.extract_many(blocks)
        except Exception as exc:
            if isinstance(exc, EvoRAGProcessorError):
                raise
            raise EvoRAGProcessorError(str(exc)) from exc

        return EvoRAGPreprocessResult(input_text=normalized_text, blocks=extractions)


async def preprocess_text_async(text: str, *, config: EvoRAGSettings = settings) -> EvoRAGPreprocessResult:
    processor = EvoRAGProcessor(config=config)
    return await processor.preprocess(text)


def preprocess_text(text: str, *, config: EvoRAGSettings = settings) -> EvoRAGPreprocessResult:
    return asyncio.run(preprocess_text_async(text, config=config))


def result_to_dict(result: EvoRAGPreprocessResult) -> dict[str, Any]:
    return result.model_dump()
