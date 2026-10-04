"""
config.py — Centralized Application Configuration
AI Research Paper Assistant

All settings are read from environment variables or a .env file.
This is the SINGLE source of truth for every configurable value.

Why centralized config?
  - Change one value → affects the entire application
  - Never hardcode secrets in source code
  - Different values per environment (dev / staging / prod)
    via different .env files or environment variables

Usage (in any other file):
  from config import settings
  settings.llm_model          → "gpt-4o-mini"
  settings.openai_api_key     → "sk-..."
"""

import logging
from typing import List
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    All values can be overridden by:
      1. Environment variables (highest priority)
      2. .env file
      3. Default values here (lowest priority)

    Pydantic automatically validates types.
    Wrong type → raises ValidationError at startup (fail fast).
    """

    # ── OpenAI ───────────────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key. NEVER hardcode. Must be set in .env or environment."
    )

    # ── Models ───────────────────────────────────────────────────────────────
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI chat model for answer generation."
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI model for generating 1536-dim text embeddings."
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Dimensionality of embedding vectors. Must match embedding_model."
    )

    # ── LLM Generation Parameters ────────────────────────────────────────────
    llm_temperature: float = Field(
        default=0.2,
        ge=0.0, le=2.0,
        description="Lower = more factual. 0.2 is good for RAG."
    )
    llm_max_tokens: int = Field(
        default=1500,
        ge=100, le=4096,
        description="Max output tokens per LLM response."
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_db_path: str = Field(
        default="./chroma_db",
        description="Local path where ChromaDB persists vector data."
    )
    chroma_collection_name: str = Field(
        default="research_papers",
        description="Name of the ChromaDB collection used to store paper embeddings."
    )

    # ── File Storage ─────────────────────────────────────────────────────────
    upload_dir: str = Field(
        default="./uploads",
        description="Local directory where uploaded PDF files are saved."
    )
    max_file_size_mb: int = Field(
        default=50,
        ge=1, le=500,
        description="Maximum PDF upload size in megabytes."
    )

    # ── RAG Retrieval ─────────────────────────────────────────────────────────
    top_k_results: int = Field(
        default=5,
        ge=1, le=20,
        description="Number of chunks retrieved from ChromaDB per query."
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=500,
        ge=100, le=4000,
        description="Max characters per text chunk (character-based, not token-based)."
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0, le=500,
        description="Character overlap between consecutive chunks to preserve context."
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins."
    )

    # ── App Meta ─────────────────────────────────────────────────────────────
    app_env: str = Field(
        default="development",
        description="Environment name: development | staging | production"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG | INFO | WARNING | ERROR"
    )

    # ─── Pydantic Settings Config ─────────────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=".env",          # read from .env file
        env_file_encoding="utf-8",
        extra="ignore",           # silently ignore unknown env vars
        case_sensitive=False,     # OPENAI_API_KEY = openai_api_key
    )

    # ─── Validators ──────────────────────────────────────────────────────────
    @field_validator("openai_api_key")
    @classmethod
    def warn_if_no_api_key(cls, v: str) -> str:
        """
        Warn loudly at startup if API key is missing.
        Don't crash (may be running tests) but make it very visible.
        """
        if not v or not v.strip():
            logger.warning(
                "⚠️  OPENAI_API_KEY is not set! "
                "LLM and embedding calls will fail. "
                "Set it in your .env file."
            )
        elif not v.startswith("sk-"):
            logger.warning(
                "⚠️  OPENAI_API_KEY does not look like a valid OpenAI key "
                "(expected 'sk-' prefix). Verify your .env file."
            )
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        """Overlap must be smaller than chunk_size, otherwise chunks are redundant."""
        chunk_size = info.data.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})."
            )
        return v

    @field_validator("app_env")
    @classmethod
    def valid_environment(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"app_env must be one of {allowed}, got '{v}'")
        return v.lower()

    # ─── Computed Properties ──────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> List[str]:
        """
        Split the comma-separated cors_origins string into a list.
        "http://localhost:3000, https://myapp.com"
        → ["http://localhost:3000", "https://myapp.com"]
        """
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB → bytes for size comparisons."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Quick check for production-only logic."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


# ─── Singleton ────────────────────────────────────────────────────────────────
# Instantiated ONCE at import time.
# Every file does: from config import settings
# They ALL share the exact same object.
settings = Settings()
