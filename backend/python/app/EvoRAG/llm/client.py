import asyncio
import json
from typing import Any

from openai import AsyncOpenAI

from app.EvoRAG.config import EvoRAGSettings, settings
from app.EvoRAG.llm.circuit_breaker import CircuitBreaker
from app.EvoRAG.llm.retry import retry_async


class EvoRAGLLMError(RuntimeError):
    pass


class EvoRAGLLMClient:
    def __init__(self, config: EvoRAGSettings = settings) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrency))
        self._circuit = CircuitBreaker(
            failure_threshold=config.circuit_failure_threshold,
            cooldown_seconds=config.circuit_cooldown_seconds,
        )
        self._client = AsyncOpenAI(
            api_key=config.api_key or "missing-evorag-api-key",
            base_url=config.api_base_url or None,
            timeout=config.timeout_seconds,
        )

    async def chat_json(self, *, system_prompt: str, user_payload: dict[str, Any], operation_name: str) -> dict[str, Any]:
        self._validate_config()
        request_text = json.dumps(user_payload, ensure_ascii=False, indent=2)

        async def operation() -> dict[str, Any]:
            async with self._semaphore:
                self._circuit.before_call()
                response = await self._client.chat.completions.create(
                    model=self.config.inference_model,
                    temperature=self.config.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": request_text},
                    ],
                )
                return parse_json_response(response.choices[0].message.content, operation_name)

        try:
            return await retry_async(
                operation,
                operation_name=operation_name,
                attempts=self.config.retry_attempts,
                base_delay=self.config.retry_base_delay_seconds,
                max_delay=self.config.retry_max_delay_seconds,
                circuit=self._circuit,
            )
        except Exception as exc:
            if isinstance(exc, EvoRAGLLMError):
                raise
            raise EvoRAGLLMError(str(exc)) from exc

    def _validate_config(self) -> None:
        if not self.config.api_key or self.config.api_key in {"change-me", "your-api-key"}:
            raise EvoRAGLLMError("EVORAG_API_KEY is not configured")
        if not self.config.inference_model:
            raise EvoRAGLLMError("EVORAG_INFERENCE_MODEL is not configured")


def parse_json_response(content: str | None, operation_name: str) -> dict[str, Any]:
    try:
        data = json.loads(content or "{}")
    except json.JSONDecodeError as exc:
        raise EvoRAGLLMError(f"{operation_name} returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise EvoRAGLLMError(f"{operation_name} returned non-object JSON")

    return data
