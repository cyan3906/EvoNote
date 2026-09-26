import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.EvoRAG.llm.circuit_breaker import CircuitBreaker, CircuitOpenError


T = TypeVar("T")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    operation_name: str,
    attempts: int,
    base_delay: float,
    max_delay: float,
    circuit: CircuitBreaker,
) -> T:
    normalized_attempts = max(1, attempts)
    delay = max(0.0, base_delay)
    max_delay = max(delay, max_delay)
    last_error: Exception | None = None

    for attempt in range(1, normalized_attempts + 1):
        try:
            result = await operation()
            circuit.record_success()
            return result
        except CircuitOpenError:
            raise
        except Exception as exc:
            last_error = exc
            circuit.record_failure()

            if attempt >= normalized_attempts:
                break

            if delay > 0:
                await asyncio.sleep(min(delay, max_delay))
                delay = min(delay * 2, max_delay)

    raise RuntimeError(f"{operation_name} failed after {normalized_attempts} attempt(s): {last_error}") from last_error
