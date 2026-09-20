"""Typed configuration loaded from environment variables (prefix ``APP_``).

Only ``.env.example`` is committed. Secrets are ``SecretStr`` so they never appear in ``repr``
or logs. API keys are configured as SHA-256 digests, so the process never holds raw keys; the
three digest lists correspond to the ``ingress``, ``admin`` and ``fault`` scopes.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ORIGIN = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


def _check_digests(value: list[str]) -> list[str]:
    for digest in value:
        if not _SHA256_HEX.fullmatch(digest):
            raise ValueError("API key digests must be lowercase hex SHA-256 digests")
    return value


class Settings(BaseSettings):
    """Service settings. Every field can be set with ``APP_<FIELD_NAME>``."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- service ---
    service_name: str = Field(
        default="agent-supervision-tree", pattern=r"^[a-z0-9][a-z0-9-]{0,62}$"
    )
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_docs: bool = False

    # --- HTTP security ---
    api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    admin_api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    fault_api_key_sha256: Annotated[list[str], NoDecode] = Field(default_factory=list)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    rate_limit_per_second: float = Field(default=5.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)
    rate_limit_max_clients: int = Field(default=10_000, ge=1)
    max_body_bytes: int = Field(default=1_048_576, ge=1)
    metrics_require_auth: bool = True

    # --- observability ---
    tracing_exporter: Literal["none", "console"] = "none"
    loop_lag_threshold_seconds: float = Field(default=0.5, gt=0)

    # --- LLM provider ---
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "anthropic_api_key")
    )
    llm_model: str = "claude-opus-5"
    llm_extractor_model: str = "claude-sonnet-5"
    llm_fallback_model: str = "claude-haiku-4-5"
    llm_allow_defer: bool = True
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # The SDK default. Non-streaming requests up to 16K output tokens can legitimately take minutes;
    # a shorter total timeout cuts them off, and each retry can be billed again.
    llm_timeout_seconds: float = Field(default=600.0, gt=0)
    llm_refusal_fallback: bool = True
    fault_rate: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- supervision tree ---
    workers: int = Field(default=4, ge=1, le=64)
    mailbox_capacity: int = Field(default=64, ge=1, le=100_000)
    worker_mailbox_capacity: int = Field(default=8, ge=1, le=10_000)
    max_message_bytes: int = Field(default=65_536, ge=256)
    message_ttl_seconds: float = Field(default=10.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=20)
    max_restarts: int = Field(default=5, ge=0)
    max_restart_seconds: float = Field(default=5.0, gt=0)
    provider_call_timeout_seconds: float = Field(default=2.0, gt=0)
    provider_bulkhead: int = Field(default=8, ge=1, le=256)
    quarantine_max_records: int = Field(default=10_000, ge=1)
    budget_max_tokens: int = Field(default=2_000_000, ge=0)
    budget_max_usd: float = Field(default=5.0, ge=0)
    webhook_url: str = Field(
        default="https://hooks.example.internal/invoices",
        pattern=r"^https://[A-Za-z0-9.-]+(/[A-Za-z0-9._/-]*)?$",
    )
    faults_enabled: bool = False

    # --- storage ---
    data_dir: Path = Path("var")
    # Where the runtime writes its hash-chained audit log. Unset keeps the chain in memory only,
    # which is what the container image does (its root filesystem is read-only).
    audit_log_path: Path | None = None

    @field_validator(
        "api_key_sha256", "admin_api_key_sha256", "fault_api_key_sha256", mode="before"
    )
    @classmethod
    def _parse_key_hashes(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("api_key_sha256", "admin_api_key_sha256", "fault_api_key_sha256")
    @classmethod
    def _check_key_hashes(cls, value: list[str]) -> list[str]:
        return _check_digests(value)

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        return _split_csv(value)

    @field_validator("cors_allow_origins")
    @classmethod
    def _check_origins(cls, value: list[str]) -> list[str]:
        for origin in value:
            if not _ORIGIN.fullmatch(origin):
                raise ValueError(
                    "cors_allow_origins must be explicit scheme://host[:port] origins "
                    "('*' is refused)"
                )
        return value

    @property
    def budget_max_usd_micros(self) -> int:
        return round(self.budget_max_usd * 1_000_000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (cached). Tests construct ``Settings(...)`` directly instead."""
    return Settings()
