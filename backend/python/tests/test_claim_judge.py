from app.core import evolution


class FakeMessage:
    content = '{"relation":"conflict","confidence":0.91,"risk_level":"high","reason":"两条 claim 对是否需要 TCP 存在明确冲突。"}'


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]
    usage = None


class FakeCompletions:
    def create(self, **kwargs):
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["response_format"] == {"type": "json_object"}
        return FakeResponse()


class FakeChat:
    completions = FakeCompletions()


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.chat = FakeChat()


def test_claim_relation_judge_uses_configured_model(monkeypatch) -> None:
    monkeypatch.setattr(evolution, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(evolution.settings, "claim_judge_api_key", "test-key")
    monkeypatch.setattr(evolution.settings, "claim_judge_api_base_url", "https://example.test/v1")
    monkeypatch.setattr(evolution.settings, "claim_judge_model", "gpt-4o")

    result = evolution.judge_claim_relation_with_model(
        {
            "claim_text": "HTTPS 通常基于 TCP 传输。",
            "source_text": "HTTPS 通常运行在 TCP 之上。",
            "keywords": ["HTTPS", "TCP"],
        },
        {
            "claim_text": "HTTPS 不需要 TCP。",
            "source_text": "HTTPS 不需要 TCP。",
            "keywords": ["HTTPS", "TCP"],
        },
        "conflict",
        0.62,
    )

    assert result["relation"] == "conflict"
    assert result["confidence"] == 0.91
    assert result["risk_level"] == "high"
