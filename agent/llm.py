"""
CareerPilot — LLM Factory (Phase 2)

Provides a cached singleton ChatGoogleGenerativeAI instance.
All agent nodes import from here — dynamically resolves Gemini API key.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger

from backend.config import get_settings


def get_effective_gemini_api_key() -> str:
    """Resolve the Gemini API key from environment, settings, or Streamlit secrets."""
    key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if key:
        return key

    settings = get_settings()
    if settings.gemini_api_key and settings.gemini_api_key.strip():
        return settings.gemini_api_key.strip()

    st = sys.modules.get("streamlit")
    if st and hasattr(st, "secrets"):
        try:
            if "GEMINI_API_KEY" in st.secrets and str(st.secrets["GEMINI_API_KEY"]).strip():
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets and str(st.secrets["GOOGLE_API_KEY"]).strip():
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
        except Exception:
            pass

    return ""


@lru_cache(maxsize=8)
def _create_llm_instance(
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> ChatGoogleGenerativeAI:
    logger.info(f"Initialising LLM: {model}")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key or None,
        temperature=temperature,
        max_output_tokens=max_tokens,
        convert_system_message_to_human=False,
    )


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return a cached Gemini LLM instance configured from settings.
    Dynamically resolves API key so late secret updates or user sidebar inputs work immediately.
    """
    settings = get_settings()
    api_key = get_effective_gemini_api_key()

    if not api_key:
        if settings.is_production:
            raise ValueError(
                "CRITICAL: GEMINI_API_KEY is not configured in production. "
                "Set GEMINI_API_KEY in your environment or secrets."
            )
        logger.warning(
            "GEMINI_API_KEY is not set; LLM invocation will fail unless running under mock testing."
        )

    return _create_llm_instance(
        settings.llm_model,
        api_key,
        settings.llm_temperature,
        settings.llm_max_tokens,
    )
