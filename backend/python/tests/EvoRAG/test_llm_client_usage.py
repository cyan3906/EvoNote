import asyncio
from types import SimpleNamespace

from app.EvoRAG.config import EvoRAGSettings
from app.EvoRAG.llm.client import EvoRAGLLMClient


def test_chat_json_records_usage_and_duration() -> None:
    completions = FakeCompletions()
    config = EvoRAGSettings(api_key="test-key", inference_model="reasoner", retry_attempts=1)
    client = EvoRAGLLMClient(config)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    checkpoint = client.usage_checkpoint()
    data = asyncio.run(
        client.chat_json(
            system_prompt="Return JSON.",
            user_payload={"text": "hello"},
            operation_name="usage test",
        )
    )
    records = client.usage_records_since(checkpoint)

    assert data == {"ok": True}
    assert completions.kwargs["model"] == "reasoner"
    assert len(records) == 1
    assert records[0]["operation_name"] == "usage test"
    assert records[0]["model"] == "reasoner"
    assert records[0]["prompt_tokens"] == 7
    assert records[0]["completion_tokens"] == 3
    assert records[0]["total_tokens"] == 10
    assert records[0]["duration_ms"] >= 0


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = {}

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=3,
                total_tokens=10,
            ),
        )
