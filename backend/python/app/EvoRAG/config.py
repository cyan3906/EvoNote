from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


EVORAG_DIR = Path(__file__).resolve().parent
ENV_FILE = EVORAG_DIR / ".env"
PROJECT_DATA_DIR = EVORAG_DIR.parents[1] / "data"


class EvoRAGSettings(BaseSettings):
    api_base_url: str = ""
    api_key: str = ""
    inference_model: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 3072
    timeout_seconds: int = 45
    temperature: float = 0

    max_concurrency: int = 4
    retry_attempts: int = 3
    retry_base_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 6.0
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: float = 30.0

    max_blocks: int = 32
    max_block_chars: int = 1800

    default_workspace_id: str = "local"
    default_project_id: str = "evorag"
    default_collection_id: str = "default"
    default_domain: str = "general"

    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "evorag"
    mysql_charset: str = "utf8mb4"

    entity_resolution_top_k: int = 3
    entity_resolution_auto_match_threshold: float = 0.9
    entity_resolution_llm_threshold: float = 0.7
    entity_resolution_manual_threshold: float = 0.6
    entity_resolution_rrf_k: int = 60
    entity_merge_max_concurrency: int = 4
    entity_vector_dimensions: int = 3072

    es_url: str = "http://127.0.0.1:9200"
    es_entity_index: str = "evorag_entities"

    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_token: str = ""
    milvus_entity_collection: str = "evorag_entities"

    milvus_data_dir: str = str(PROJECT_DATA_DIR / "milvus")
    elasticsearch_data_dir: str = str(PROJECT_DATA_DIR / "elasticsearch")

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="EVORAG_",
        extra="ignore",
    )


settings = EvoRAGSettings()
