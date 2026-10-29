"""
CareerPilot — Application Configuration
Loads all settings from environment variables (via .env).
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent

# ── Sync Streamlit Secrets (for Streamlit Cloud deployments) ──────────────────
if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_PORT"):
    try:
        import streamlit as _st
        if hasattr(_st, "secrets"):
            for _k, _v in _st.secrets.items():
                if isinstance(_v, dict):
                    for _sk, _sv in _v.items():
                        if _sk not in os.environ:
                            os.environ[_sk] = str(_sv)
                elif _k not in os.environ:
                    os.environ[_k] = str(_v)
    except Exception:
        pass


class Settings(BaseSettings):
    """All application settings, loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── LLM ─────────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8192

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = "models/gemini-embedding-001"

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: str = str(ROOT_DIR / "chroma_db")
    chroma_collection_name: str = "careerpilot_docs"

    # ── SQLite ───────────────────────────────────────────────────────────────
    database_url: str = f"sqlite:///{ROOT_DIR / 'database' / 'careerpilot.db'}"

    # ── RAG ─────────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 5

    # ── Streamlit & CORS ─────────────────────────────────────────────────────
    streamlit_port: int = 8501
    cors_allowed_origins: list[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://testserver",
        "testserver",
    ]

    # ── Security & Sessions ──────────────────────────────────────────────────
    auth_secret_key: str = "careerpilot-session-signing-secret-key-2026"
    token_expiry_hours: int = 168

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def documents_dir(self) -> Path:
        return ROOT_DIR / "data" / "documents"

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"

    def __getattribute__(self, name: str) -> Any:
        if name == "gemini_api_key":
            env_val = (
                os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_API_KEY", "").strip()
            )
            if env_val:
                return env_val
            st = sys.modules.get("streamlit")
            if st and hasattr(st, "secrets"):
                try:
                    if "GEMINI_API_KEY" in st.secrets and str(st.secrets["GEMINI_API_KEY"]).strip():
                        return str(st.secrets["GEMINI_API_KEY"]).strip()
                    if "GOOGLE_API_KEY" in st.secrets and str(st.secrets["GOOGLE_API_KEY"]).strip():
                        return str(st.secrets["GOOGLE_API_KEY"]).strip()
                except Exception:
                    pass
        return super().__getattribute__(name)

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in ("production", "prod")

    def validate_required(self) -> None:
        """Raise ValueError if mandatory secrets are missing."""
        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )

    def validate_startup(self) -> None:
        """Strict startup validation for production environments."""
        if self.is_production:
            if not self.gemini_api_key:
                raise ValueError(
                    "CRITICAL: GEMINI_API_KEY is not set in production. "
                    "Application cannot boot safely without LLM credentials."
                )
            if "*" in self.cors_allowed_origins:
                raise ValueError(
                    "CRITICAL: Wildcard '*' CORS origin is forbidden in production mode. "
                    "Configure explicit origins in CORS_ALLOWED_ORIGINS."
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
