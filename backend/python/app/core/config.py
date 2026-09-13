from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Evonote API"
    app_env: str = "development"
    app_debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    agent_api_base_url: str = "https://api.uiuihao.com/v1" 
    agent_api_key: str = "sk-EvGLrMBijuYrWTRfHSYEr5qGATNAfPBJ1q1l9WtwXHyzQ3ee"
    agent_model: str = "gpt-4o-mini"
    agent_timeout_seconds: int = 30

    evolution_api_base_url: str = ""
    evolution_api_key: str = ""
    evolution_model: str = ""
    evolution_timeout_seconds: int = 0
    evolution_temperature: float = 0

    embedding_api_key: str = "sk-RpWp5bSSc8tbV9WzS26lo9qNauXL3rmFM8Nu0oqNwgJx7GtJ"
    embedding_base_url: str = "https://api.quickrouter.ai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072

    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_token: str = ""
    milvus_collection: str = "evonote_claims"

    es_url: str = "http://127.0.0.1:9200"
    es_index: str = "evonote_claims"

    retrieval_top_k: int = 20
    rrf_k: int = 60
    external_retry_attempts: int = 3
    external_retry_base_delay_seconds: float = 0.5
    external_retry_max_delay_seconds: float = 4.0

    auth_password: str = "evonote2026"
    auth_token: str = "evonote-dev-token"

    sqlite_database_path: str = "data/evonote.sqlite3"

    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_url: str = "redis://127.0.0.1:6379/0"
    redis_cache_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
