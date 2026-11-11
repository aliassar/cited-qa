"""
Application configuration.

WHY A SETTINGS OBJECT INSTEAD OF os.environ EVERYWHERE
------------------------------------------------------
1. It is *typed*. `settings.access_token_ttl_seconds` is an int, not a string
   you have to remember to cast at every call site.
2. It *fails fast*. If a required variable is missing or malformed the process
   crashes at import time with a readable Pydantic error, instead of blowing up
   at 3am on the one code path that reads it.
3. It is *testable*. Tests build their own Settings instance and override the
   dependency, rather than mutating global environment state.

This is the "config" leg of the twelve-factor app: config lives in the
environment, never in the code, and never in git.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `env_file` is a developer convenience only. In production (Docker, k8s,
    # ECS) you inject real environment variables and there is no .env file.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated env vars instead of crashing
    )

    # ---------------------------------------------------------------- app ---
    app_name: str = "cited-qa"
    environment: Literal["local", "test", "staging", "production"] = "local"
    debug: bool = False

    # `root_path` matters when you sit behind a reverse proxy that strips a
    # prefix (e.g. an ALB routing /api/* to this service). Without it, the
    # OpenAPI docs generate wrong URLs.
    root_path: str = ""

    # ----------------------------------------------------------- database ---
    # Async driver URLs. SQLite by default so the project runs with zero
    # infrastructure; docker-compose overrides this with Postgres.
    #   sqlite+aiosqlite:///./cited_qa.db
    #   postgresql+asyncpg://user:pass@localhost:5432/cited_qa
    database_url: str = "sqlite+aiosqlite:///./cited_qa.db"

    # Connection pool sizing:
    # total DB connections = pool_size * number_of_uvicorn_workers.
    # Postgres defaults to max_connections=100, so 4 workers x pool_size=20
    # (+ overflow) will exhaust the server. Size the pool from the DB budget
    # backwards, or put PgBouncer in front.
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout_seconds: int = 30
    db_echo: bool = False  # log every SQL statement; noisy; useful when debugging

    # ---------------------------------------------------------- security ---
    # NEVER ship a default secret to production. The validator below refuses to
    # boot in production if this is still the placeholder.
    # noqa S105: this is a placeholder default, and the validator below refuses
    # to let it reach production. Silencing the rule *narrowly*, with a reason.
    jwt_secret: str = "dev-only-insecure-secret-change-me"  # noqa: S105
    jwt_algorithm: str = "HS256"
    access_token_ttl_seconds: int = 15 * 60  # short: limits blast radius if leaked
    refresh_token_ttl_seconds: int = (
        14 * 24 * 3600
    )  # long: kept in an httpOnly cookie / secure store

    # bcrypt work factor. Higher = slower = harder to brute force. 12 is a
    # sane 2026 default (~250ms on a laptop). It is a *deliberate* cost.
    bcrypt_rounds: int = 12

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # ------------------------------------------------------- rate limiting ---
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 120  # tokens in the bucket
    rate_limit_window_seconds: int = 60  # refill window

    # ------------------------------------------------------------- uploads ---
    max_upload_bytes: int = 5 * 1024 * 1024  # 5 MiB
    # Plain text only: there is no PDF/DOCX extractor in this codebase (see
    # services/documents.py::extract_text), so accepting those types would only
    # defer the failure.
    allowed_upload_types: list[str] = Field(default_factory=lambda: ["text/plain", "text/markdown"])

    # ----------------------------------------------------------------- llm ---
    # "fake" = deterministic local stub (used by tests and by anyone without an
    # API key). "anthropic" = the real Claude API.
    llm_provider: Literal["fake", "anthropic"] = "fake"
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5-5"
    llm_max_tokens: int = 4096

    # ------------------------------------------------------------ workers ---
    # Number of in-process background ingestion workers. See services/ingestion.py
    # for why this is a stepping stone to a real queue (Celery / arq / RabbitMQ).
    ingest_worker_count: int = 2
    ingest_queue_maxsize: int = 1000

    @field_validator("jwt_secret")
    @classmethod
    def _reject_default_secret_in_prod(cls, v: str, info) -> str:
        # `info.data` holds the fields validated *before* this one. Field order
        # in the class body therefore matters: `environment` is declared above.
        if info.data.get("environment") == "production" and "dev-only" in v:
            raise ValueError("jwt_secret must be set to a real secret in production")
        return v

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


# lru_cache turns this into a singleton: the Settings object is built once and
# reused. FastAPI dependencies can then do `settings: Settings = Depends(get_settings)`
# and tests can call `get_settings.cache_clear()` or override the dependency.
@lru_cache
def get_settings() -> Settings:
    return Settings()
