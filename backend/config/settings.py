from functools import lru_cache
from typing import List

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = "Web Context Assistant API"
    debug: bool = False

    # LLM Provider
    llm_provider: str = "openai"  # "openai" or "anthropic"

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "AWS_GATEWAY_URL")
    )
    pat_token: str = Field(
        default="",
        validation_alias=AliasChoices("PAT_TOKEN", "OPENAI_PAT")
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
