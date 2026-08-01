from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_database_url(url: str) -> str:
    """Accept Neon/Railway postgres URLs and make them asyncpg-compatible."""
    u = url.strip()
    if u.startswith("postgres://"):
        u = "postgresql://" + u[len("postgres://") :]
    if u.startswith("postgresql://") and "+asyncpg" not in u.split("://", 1)[0]:
        u = "postgresql+asyncpg://" + u[len("postgresql://") :]
    # asyncpg uses `ssl=`, not libpq's `sslmode=`
    u = u.replace("sslmode=", "ssl=")
    return u


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+asyncpg://hrmind:hrmind@localhost:5433/hrmind"
    redis_url: str = "redis://localhost:6379/0"
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dims: int = 1536
    router_threshold: float = 0.75
    top_k: int = 30
    rerank_top_k: int = 5
    # Hybrid retrieval: RRF constant and the trigram floor for fuzzy name matching.
    retrieval_rrf_k: int = 60
    retrieval_name_similarity: float = 0.35
    max_history: int = 20
    max_context: int = 8
    summary_trigger: int = 12
    max_ids_in_packet: int = 20
    max_entity_memory: int = 50
    max_summary_chars: int = 2000
    tool_fact_ttl_seconds: int = 180
    sql_cache_ttl: int = 60
    retrieval_cache_ttl: int = 1800
    plan_cache_ttl: int = 900
    max_plan_nodes: int = 12
    tool_timeout_ms: int = 15000
    sql_max_rows: int = 200
    default_tenant_id: str = "00000000-0000-0000-0000-000000000001"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    resume_storage_dir: str = "data/seed/resumes"
    parse_version: str = "1"
    cors_origins: str = "*"
    # When true (default in production/staging), Redis is required for chat sessions.
    require_redis: bool | None = None
    session_ttl_seconds: int = 86400

    @field_validator("database_url", mode="before")
    @classmethod
    def _normalize_database_url(cls, value: object) -> object:
        if isinstance(value, str) and value:
            return normalize_database_url(value)
        return value

    @property
    def redis_required(self) -> bool:
        if self.require_redis is not None:
            return self.require_redis
        return self.app_env.lower() in {"production", "prod", "staging"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
