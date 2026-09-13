from collections.abc import Callable
from time import sleep
from typing import TypeVar

from app.core.config import settings

T = TypeVar("T")


def retry_call(operation: Callable[[], T], *, operation_name: str) -> T:
    attempts = max(1, settings.external_retry_attempts)
    delay = max(0.0, settings.external_retry_base_delay_seconds)
    max_delay = max(delay, settings.external_retry_max_delay_seconds)
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc

            if attempt >= attempts:
                break

            if delay > 0:
                sleep(min(delay, max_delay))
                delay = min(delay * 2, max_delay)

    raise RuntimeError(f"{operation_name} failed after {attempts} attempt(s): {last_error}") from last_error
