import time
from dataclasses import dataclass


class CircuitOpenError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    failure_threshold: int
    cooldown_seconds: float
    failure_count: int = 0
    opened_at: float | None = None

    def before_call(self) -> None:
        if self.opened_at is None:
            return

        elapsed = time.monotonic() - self.opened_at
        if elapsed >= self.cooldown_seconds:
            self.opened_at = None
            self.failure_count = 0
            return

        remaining = max(0.0, self.cooldown_seconds - elapsed)
        raise CircuitOpenError(f"EvoRAG LLM circuit is open, retry after {remaining:.1f}s")

    def record_success(self) -> None:
        self.failure_count = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= max(1, self.failure_threshold):
            self.opened_at = time.monotonic()
