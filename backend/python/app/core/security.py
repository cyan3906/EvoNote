import base64
import hashlib
import hmac
import secrets
import time
from secrets import compare_digest

from fastapi import Header, HTTPException, status

from app.core.config import settings

DEFAULT_AUTH_PASSWORD = "evonote2026"
DEFAULT_AUTH_TOKEN = "evonote-dev-token"
TOKEN_VERSION = "v1"


class SecurityConfigError(RuntimeError):
    pass


def validate_security_settings() -> None:
    if not _should_enforce_secure_config():
        return

    if settings.auth_password == DEFAULT_AUTH_PASSWORD and not settings.auth_password_hash:
        raise SecurityConfigError("生产环境必须配置 AUTH_PASSWORD_HASH 或修改默认 AUTH_PASSWORD")

    if settings.auth_token in {"", DEFAULT_AUTH_TOKEN, "replace-with-a-long-random-token"}:
        raise SecurityConfigError("生产环境必须配置随机 AUTH_TOKEN")

    if not settings.cors_allowed_origins.strip():
        raise SecurityConfigError("生产环境必须配置 CORS_ALLOWED_ORIGINS")


def _should_enforce_secure_config() -> bool:
    return settings.enforce_secure_config or settings.app_env.lower() in {"production", "prod"}


def verify_password(password: str) -> bool:
    if settings.auth_password_hash:
        return verify_password_hash(password, settings.auth_password_hash)

    return compare_digest(password, settings.auth_password)


def hash_password(password: str, *, iterations: int = 260000) -> str:
    salt = secrets.token_urlsafe(18)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"pbkdf2_sha256${iterations}${salt}${encoded}"


def verify_password_hash(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt, expected = password_hash.split("$", 3)
        iterations = int(iterations_text)
    except ValueError:
        return False

    if algorithm != "pbkdf2_sha256" or iterations <= 0 or not salt or not expected:
        return False

    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    actual = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return compare_digest(actual, expected)


def create_access_token() -> str:
    expires_at = int(time.time()) + max(60, int(settings.auth_token_ttl_seconds))
    nonce = secrets.token_urlsafe(12)
    payload = f"{TOKEN_VERSION}.{expires_at}.{nonce}"
    signature = _sign(payload)
    return f"{payload}.{signature}"


def verify_access_token(token: str) -> bool:
    parts = token.split(".")

    if len(parts) != 4 or parts[0] != TOKEN_VERSION:
        return _allow_development_static_token(token)

    payload = ".".join(parts[:3])
    signature = parts[3]

    try:
        expires_at = int(parts[1])
    except ValueError:
        return False

    if expires_at < int(time.time()):
        return False

    return compare_digest(signature, _sign(payload))


def _allow_development_static_token(token: str) -> bool:
    if settings.app_env.lower() not in {"development", "dev", "local", "test"}:
        return False

    return compare_digest(token, settings.auth_token)


def _sign(payload: str) -> str:
    digest = hmac.new(settings.auth_token.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="请先登录",
        )

    token = authorization.removeprefix("Bearer ").strip()

    if not verify_access_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已失效",
        )
