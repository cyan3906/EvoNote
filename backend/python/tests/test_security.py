from app.core import security


def test_password_hash_verification() -> None:
    password_hash = security.hash_password("secret-password")

    assert security.verify_password_hash("secret-password", password_hash)
    assert not security.verify_password_hash("wrong-password", password_hash)


def test_access_token_expires_and_verifies(monkeypatch) -> None:
    monkeypatch.setattr(security.settings, "auth_token", "test-secret-token")
    monkeypatch.setattr(security.settings, "auth_token_ttl_seconds", 60)

    token = security.create_access_token()

    assert security.verify_access_token(token)
    assert not security.verify_access_token(f"{token}x")
