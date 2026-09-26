import asyncio
from time import perf_counter
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
        total_started_at = perf_counter()
        timings: dict[str, float] = {}
        normalized_text = str(text or "").strip()
        if not normalized_text:
            raise EvoRAGProcessorError("input text is empty")

        try:
            before_block_split_at = perf_counter()
            timings["startup_to_first_block_split_ms"] = elapsed_ms(total_started_at)
            if hasattr(self.block_splitter, "split_with_timings"):
                blocks, split_timings = await self.block_splitter.split_with_timings(normalized_text)
                timings.update(split_timings)
            else:
                blocks = await self.block_splitter.split(normalized_text)
                timings["entity_anchor_split_ms"] = elapsed_ms(before_block_split_at)
                timings["physical_chunk_split_ms"] = 0.0
            entity_extraction_started_at = perf_counter()
            extractions = await self.entity_extractor.extract_many(blocks)
            timings["entity_extraction_ms"] = elapsed_ms(entity_extraction_started_at)
        except Exception as exc:
            if isinstance(exc, EvoRAGProcessorError):
                raise
            raise EvoRAGProcessorError(str(exc)) from exc

        timings["total_ms"] = elapsed_ms(total_started_at)
        return EvoRAGPreprocessResult(input_text=normalized_text, blocks=extractions, timings=timings)


async def preprocess_text_async(text: str, *, config: EvoRAGSettings = settings) -> EvoRAGPreprocessResult:
    processor = EvoRAGProcessor(config=config)
    return await processor.preprocess(text)


def preprocess_text(text: str, *, config: EvoRAGSettings = settings) -> EvoRAGPreprocessResult:
    return asyncio.run(preprocess_text_async(text, config=config))


def result_to_dict(result: EvoRAGPreprocessResult) -> dict[str, Any]:
    return result.model_dump()


def elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)
