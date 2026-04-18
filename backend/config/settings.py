import logging
from functools import lru_cache
from typing import List

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "Web Context Assistant API"
    debug: bool = False

    # LLM Provider
    llm_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices("LLM_PROVIDER", "MODEL_PROVIDER")
    )  # "openai", "anthropic", "ollama", or "hybrid"
    hybrid_cloud_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices("HYBRID_CLOUD_PROVIDER")
    )
    hybrid_prefilter_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("HYBRID_PREFILTER_ENABLED")
    )
    hybrid_prefilter_max_chunks: int = Field(
        default=4,
        validation_alias=AliasChoices("HYBRID_PREFILTER_MAX_CHUNKS")
    )
    hybrid_prefilter_max_chars_per_chunk: int = Field(
        default=420,
        validation_alias=AliasChoices("HYBRID_PREFILTER_MAX_CHARS_PER_CHUNK")
    )
    hybrid_prefilter_timeout_seconds: int = Field(
        default=20,
        validation_alias=AliasChoices("HYBRID_PREFILTER_TIMEOUT_SECONDS")
    )

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("OPENAI_MODEL", "MODEL")
    )
    openai_summary_model: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_MODEL_SUMMARY")
    )
    openai_reasoning_model: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_MODEL_REASONING")
    )
    openai_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "AWS_GATEWAY_URL")
    )
    pat_token: str = Field(
        default="",
        validation_alias=AliasChoices("PAT_TOKEN", "OPENAI_PAT")
    )
    openai_reasoning_effort: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_REASONING_EFFORT", "MODEL_REASONING_EFFORT")
    )
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL")
    )
    ollama_summary_model: str = Field(
        default="gemma2:2b",
        validation_alias=AliasChoices("OLLAMA_MODEL_SUMMARY", "OLLAMA_MODEL", "LOCAL_MODEL_SUMMARY")
    )
    ollama_reasoning_model: str = Field(
        default="deepseek-coder-v2:16b-lite-instruct",
        validation_alias=AliasChoices("OLLAMA_MODEL_REASONING", "LOCAL_MODEL_REASONING")
    )
    ollama_request_timeout_seconds: int = Field(
        default=45,
        validation_alias=AliasChoices("OLLAMA_REQUEST_TIMEOUT_SECONDS")
    )
    embedding_provider: str = Field(
        default="openai",
        validation_alias=AliasChoices("EMBEDDING_PROVIDER")
    )
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embedding_dimensions: int = 384
    local_embedding_use_subprocess: bool = True
    local_embedding_startup_timeout_seconds: int = 20
    local_embedding_inference_timeout_seconds: int = 60

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    anthropic_summary_model: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_MODEL_SUMMARY")
    )
    anthropic_reasoning_model: str = Field(
        default="",
        validation_alias=AliasChoices("ANTHROPIC_MODEL_REASONING")
    )

    # Limits
    max_input_tokens: int = 100000
    max_output_tokens: int = 4096
    request_timeout_seconds: int = 120
    log_level: str = "INFO"

    # Retrieval quality
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 12
    reranker_use_subprocess: bool = True
    reranker_startup_timeout_seconds: int = 8
    reranker_inference_timeout_seconds: int = 12

    # CORS
    cors_origins: List[str] = ["chrome-extension://*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    @field_validator("llm_provider", mode="before")
    @classmethod
    def normalize_llm_provider(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        provider_aliases = {
            "openai": "openai",
            "anthropic": "anthropic",
            "ollama": "ollama",
            "local": "ollama",
            "local-llm": "ollama",
            "local_llm": "ollama",
            "hybrid": "hybrid",
            "hybrid-local-cloud": "hybrid",
            "hybrid_local_cloud": "hybrid",
            "ly-chatai": "openai",
            "ly_chatai": "openai",
            "openai-compatible": "openai",
            "openai_compatible": "openai",
            "gateway": "openai",
        }
        return provider_aliases.get(normalized, normalized or "openai")

    @field_validator("hybrid_cloud_provider", mode="before")
    @classmethod
    def normalize_hybrid_cloud_provider(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        provider_aliases = {
            "openai": "openai",
            "anthropic": "anthropic",
            "ly-chatai": "openai",
            "ly_chatai": "openai",
            "openai-compatible": "openai",
            "openai_compatible": "openai",
            "gateway": "openai",
        }
        return provider_aliases.get(normalized, normalized or "openai")

    @field_validator("openai_reasoning_effort", mode="before")
    @classmethod
    def normalize_openai_reasoning_effort(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized == "xhigh":
            return "high"
        return normalized if normalized in {"", "low", "medium", "high"} else ""

    @field_validator("embedding_provider", mode="before")
    @classmethod
    def normalize_embedding_provider(cls, value: str) -> str:
        normalized = str(value or "").strip().lower()
        provider_aliases = {
            "openai": "openai",
            "gateway": "openai",
            "remote": "openai",
            "local": "local",
            "sentence-transformers": "local",
            "sentence_transformers": "local",
            "sbert": "local",
        }
        return provider_aliases.get(normalized, normalized or "openai")

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = str(value or "").strip().upper()
        return normalized if normalized in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"

    def resolved_openai_api_key(self) -> str:
        """Prefer internal PAT when provided, otherwise use the standard OpenAI key."""
        return self.pat_token or self.openai_api_key

    def resolved_openai_base_url(self) -> str | None:
        """Normalized OpenAI-compatible base URL, including internal gateways."""
        normalized = self.openai_base_url.strip().rstrip("/")
        return normalized or None

    def resolved_ollama_base_url(self) -> str:
        normalized = self.ollama_base_url.strip().rstrip("/")
        return normalized or "http://host.docker.internal:11434"

    def resolved_ollama_openai_base_url(self) -> str:
        base_url = self.resolved_ollama_base_url()
        return base_url if base_url.endswith("/v1") else f"{base_url}/v1"

    def resolved_llm_base_url(self) -> str | None:
        if self.llm_provider == "openai":
            return self.resolved_openai_base_url()
        if self.llm_provider == "hybrid":
            return self.resolved_openai_base_url() if self.resolved_cloud_provider() == "openai" else None
        if self.llm_provider == "ollama":
            return self.resolved_ollama_base_url()
        return None

    def resolved_cloud_provider(self) -> str:
        if self.llm_provider == "hybrid":
            return self.hybrid_cloud_provider
        return self.llm_provider

    def hybrid_mode_enabled(self) -> bool:
        return self.llm_provider == "hybrid"

    def resolved_summary_model(self) -> str:
        provider = self.resolved_cloud_provider()
        if provider == "openai":
            return self.openai_summary_model or self.openai_model
        if provider == "ollama":
            return self.ollama_summary_model
        return self.anthropic_summary_model or self.anthropic_model

    def resolved_reasoning_model(self) -> str:
        provider = self.resolved_cloud_provider()
        if provider == "openai":
            return self.openai_reasoning_model or self.resolved_summary_model()
        if provider == "ollama":
            return self.ollama_reasoning_model or self.ollama_summary_model
        return self.anthropic_reasoning_model or self.resolved_summary_model()

    def resolved_primary_model(self) -> str:
        return self.resolved_summary_model()

    def resolved_prefilter_model(self) -> str:
        return self.ollama_summary_model

    def resolved_model_route(self, task_kind: str | None) -> str:
        normalized = str(task_kind or "answer").strip().lower()
        return "reasoning" if normalized in {"critique", "diagram"} else "summary"

    def resolved_model_for_task(self, task_kind: str | None) -> str:
        return (
            self.resolved_reasoning_model()
            if self.resolved_model_route(task_kind) == "reasoning"
            else self.resolved_summary_model()
        )

    def resolved_log_level(self) -> int:
        return getattr(logging, self.log_level, logging.INFO)

    def diagnostics_summary(self) -> dict[str, object]:
        """Sanitized configuration summary for startup logs."""
        return {
            "app_name": self.app_name,
            "debug": self.debug,
            "llm_provider": self.llm_provider,
            "cloud_provider": self.resolved_cloud_provider() if self.hybrid_mode_enabled() else None,
            "model": self.resolved_summary_model(),
            "reasoning_model": self.resolved_reasoning_model(),
            "prefilter_model": self.resolved_prefilter_model() if self.hybrid_mode_enabled() else None,
            "hybrid_prefilter_enabled": self.hybrid_prefilter_enabled if self.hybrid_mode_enabled() else None,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.resolved_embedding_model(),
            "embedding_dimensions": self.resolved_embedding_dimensions(),
            "reasoning_effort": self.openai_reasoning_effort or None,
            "base_url": self.resolved_llm_base_url(),
            "has_openai_api_key": bool(self.openai_api_key),
            "has_pat_token": bool(self.pat_token),
            "request_timeout_seconds": self.request_timeout_seconds,
            "ollama_request_timeout_seconds": self.ollama_request_timeout_seconds if self.llm_provider == "ollama" else None,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "reranker_enabled": self.reranker_enabled,
            "reranker_model": self.reranker_model if self.reranker_enabled else None,
            "cors_origins": self.cors_origins,
            "log_level": self.log_level,
        }

    def resolved_embedding_model(self) -> str:
        if self.embedding_provider == "local":
            return self.local_embedding_model
        return self.openai_embedding_model

    def resolved_embedding_dimensions(self) -> int:
        if self.embedding_provider == "local":
            return self.local_embedding_dimensions
        return self.openai_embedding_dimensions


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
