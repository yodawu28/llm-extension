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
    )  # "openai" or "anthropic"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("OPENAI_MODEL", "MODEL")
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
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Limits
    max_input_tokens: int = 100000
    max_output_tokens: int = 4096
    request_timeout_seconds: int = 60

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

    def resolved_openai_api_key(self) -> str:
        """Prefer internal PAT when provided, otherwise use the standard OpenAI key."""
        return self.pat_token or self.openai_api_key

    def resolved_openai_base_url(self) -> str | None:
        """Normalized OpenAI-compatible base URL, including internal gateways."""
        normalized = self.openai_base_url.strip().rstrip("/")
        return normalized or None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
