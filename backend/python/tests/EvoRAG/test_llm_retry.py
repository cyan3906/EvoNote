import asyncio

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.llm.circuit_breaker import CircuitBreaker
from app.EvoRAG.llm import retry as retry_module
from app.EvoRAG.llm.retry import retry_async


def test_evorag_llm_retry_defaults_wait_2_4_8_16_seconds() -> None:
    config = EvoRAGSettings(_env_file=None)

    assert config.retry_attempts == 5
    assert config.retry_base_delay_seconds == 2
    assert config.retry_max_delay_seconds == 16


def test_retry_async_uses_2_4_8_16_second_backoff_from_defaults(monkeypatch) -> None:
    config = EvoRAGSettings(_env_file=None)
    sleeps: list[float] = []
    attempts = 0

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def operation() -> dict[str, bool]:
        nonlocal attempts
        attempts += 1
        if attempts < config.retry_attempts:
            raise RuntimeError("temporary api failure")
        return {"ok": True}

    monkeypatch.setattr(retry_module.asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        retry_async(
            operation,
            operation_name="retry defaults test",
            attempts=config.retry_attempts,
            base_delay=config.retry_base_delay_seconds,
            max_delay=config.retry_max_delay_seconds,
            circuit=CircuitBreaker(failure_threshold=10, cooldown_seconds=1),
        )
    )

    assert result == {"ok": True}
    assert attempts == 5
    assert sleeps == [2, 4, 8, 16]
