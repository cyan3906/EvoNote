from app.core.retry import retry_call


def test_retry_call_retries_until_success(monkeypatch) -> None:
    monkeypatch.setattr("app.core.retry.settings.external_retry_attempts", 3)
    monkeypatch.setattr("app.core.retry.settings.external_retry_base_delay_seconds", 0)
    calls = {"count": 0}

    def flaky_operation() -> str:
        calls["count"] += 1

        if calls["count"] < 3:
            raise ConnectionError("temporary failure")

        return "ok"

    assert retry_call(flaky_operation, operation_name="test operation") == "ok"
    assert calls["count"] == 3
