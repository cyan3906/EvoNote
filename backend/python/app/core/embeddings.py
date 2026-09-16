from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock
from typing import Callable

from openai import OpenAI

from app.core.config import settings

Vector = list[float]
FallbackVector = Callable[[str], Vector]


class EmbeddingCircuitOpen(RuntimeError):
    pass


@dataclass
class EmbeddingBatchStats:
    requested: int = 0
    embedded: int = 0
    fallback: int = 0
    batches: int = 0
    failed_batches: int = 0
    elapsed_ms: float = 0.0
    circuit_open: bool = False


class CircuitBreaker:
    def __init__(self, *, failure_threshold: int, cooldown_seconds: float) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self._lock = Lock()
        self._consecutive_failures = 0
        self._opened_at = 0.0

    def before_call(self) -> None:
        with self._lock:
            if not self._opened_at:
                return

            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._opened_at = 0.0
                self._consecutive_failures = 0
                return

            raise EmbeddingCircuitOpen("Embedding circuit breaker is open.")

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = 0.0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1

            if self._consecutive_failures >= self.failure_threshold:
                self._opened_at = time.monotonic()

    def is_open(self) -> bool:
        with self._lock:
            return bool(self._opened_at)


_BREAKER = CircuitBreaker(
    failure_threshold=settings.embedding_circuit_failure_threshold,
    cooldown_seconds=settings.embedding_circuit_cooldown_seconds,
)


def embed_text(text: str) -> Vector:
    vectors, _stats = embed_texts([text], fallback_vector=None)
    return vectors[0] if vectors else []


def embed_texts(
    texts: list[str],
    *,
    batch_size: int | None = None,
    max_concurrency: int | None = None,
    fallback_vector: FallbackVector | None = None,
) -> tuple[list[Vector], EmbeddingBatchStats]:
    started = time.perf_counter()
    stats = EmbeddingBatchStats(requested=len(texts))

    if not texts:
        return [], stats

    api_key = settings.embedding_api_key.strip() or settings.agent_api_key.strip()
    base_url = settings.embedding_base_url.strip() or settings.agent_api_base_url.strip()

    if not api_key:
        vectors = [_fallback(text, fallback_vector) for text in texts]
        stats.fallback = len(vectors)
        stats.elapsed_ms = _elapsed_ms(started)
        return vectors, stats

    normalized_batch_size = max(1, batch_size or settings.embedding_batch_size)
    normalized_concurrency = max(1, max_concurrency or settings.embedding_max_concurrency)
    indexed_texts = [(index, text.strip()) for index, text in enumerate(texts)]
    results: list[Vector | None] = [None] * len(texts)
    fallback_count = 0

    blank_indexes = [index for index, text in indexed_texts if not text]
    for index in blank_indexes:
        results[index] = _fallback(texts[index], fallback_vector)
        fallback_count += 1

    non_blank = [(index, text) for index, text in indexed_texts if text]
    unique_texts: dict[str, list[int]] = {}

    for index, text in non_blank:
        unique_texts.setdefault(text, []).append(index)

    unique_items = list(unique_texts.items())
    batches = [
        unique_items[offset : offset + normalized_batch_size]
        for offset in range(0, len(unique_items), normalized_batch_size)
    ]
    stats.batches = len(batches)

    if _BREAKER.is_open():
        stats.circuit_open = True
        for text, indexes in unique_items:
            vector = _fallback(text, fallback_vector)
            for index in indexes:
                results[index] = vector
                fallback_count += 1
        stats.fallback = fallback_count
        stats.elapsed_ms = _elapsed_ms(started)
        return _finalize_vectors(results), stats

    client = OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=settings.agent_timeout_seconds,
    )
    semaphore = BoundedSemaphore(normalized_concurrency)

    with ThreadPoolExecutor(max_workers=normalized_concurrency, thread_name_prefix="embedding-batch") as executor:
        futures = [
            executor.submit(_embed_batch_guarded, client, semaphore, [text for text, _indexes in batch])
            for batch in batches
        ]

        for future, batch in zip(futures, batches, strict=False):
            try:
                vectors = future.result()
            except Exception:
                stats.failed_batches += 1
                if _BREAKER.is_open():
                    stats.circuit_open = True

                for text, indexes in batch:
                    vector = _fallback(text, fallback_vector)
                    for index in indexes:
                        results[index] = vector
                        fallback_count += 1
                continue

            for (text, indexes), vector in zip(batch, vectors, strict=False):
                for index in indexes:
                    results[index] = vector

    finalized = _finalize_vectors(results)
    stats.fallback = fallback_count
    stats.embedded = max(0, len(finalized) - stats.fallback)
    stats.elapsed_ms = _elapsed_ms(started)
    return finalized, stats


def _embed_batch_guarded(client: OpenAI, semaphore: BoundedSemaphore, texts: list[str]) -> list[Vector]:
    with semaphore:
        return _embed_batch_with_retry(client, texts)


def _embed_batch_with_retry(client: OpenAI, texts: list[str]) -> list[Vector]:
    attempts = max(1, settings.external_retry_attempts)
    delay = max(0.0, settings.external_retry_base_delay_seconds)
    max_delay = max(delay, settings.external_retry_max_delay_seconds)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            _BREAKER.before_call()
            response = client.embeddings.create(
                model=settings.embedding_model,
                input=texts,
            )
            vectors = [[float(value) for value in item.embedding] for item in response.data]

            if len(vectors) != len(texts):
                raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(vectors)}.")

            _BREAKER.record_success()
            return vectors
        except Exception as exc:
            last_error = exc
            _BREAKER.record_failure()

            if isinstance(exc, EmbeddingCircuitOpen) or attempt >= attempts:
                break

            if delay > 0:
                jitter = random.uniform(0, delay * 0.25)
                time.sleep(min(delay + jitter, max_delay))
                delay = min(delay * 2, max_delay)

    raise RuntimeError(f"Embedding batch failed after {attempts} attempt(s): {last_error}") from last_error


def _fallback(text: str, fallback_vector: FallbackVector | None) -> Vector:
    return fallback_vector(text) if fallback_vector else []


def _finalize_vectors(vectors: list[Vector | None]) -> list[Vector]:
    return [vector if vector is not None else [] for vector in vectors]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
