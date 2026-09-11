from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Evonote API"
    app_env: str = "development"
    app_debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    agent_api_base_url: str = "https://api.uiuihao.com/v1" 
    agent_api_key: str = "change-me"
    agent_model: str = "default-agent"
    agent_timeout_seconds: int = 30

    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_cache_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()