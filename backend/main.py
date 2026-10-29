"""
CareerPilot — FastAPI Application Entry Point (Phase 1)
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.config import get_settings

# ── Settings ──────────────────────────────────────────────────────────────────
settings = get_settings()

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
    ),
)


from database.connection import init_db

# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("CareerPilot API starting up…")
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"LLM model   : {settings.llm_model}")
    logger.info(f"Docs        : http://{settings.app_host}:{settings.app_port}/docs")

    # 1. Startup environment validation
    settings.validate_startup()

    # 2. Database initialization - fatal error stops startup
    try:
        init_db()
        logger.info("SQLite persistent database ready.")
    except Exception as exc:
        logger.critical(f"FATAL: Failed to initialize SQLite database: {exc}")
        raise RuntimeError(f"Database initialization failed: {exc}") from exc

    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("CareerPilot API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CareerPilot",
    description=(
        "An Agentic AI for Personalized Placement Preparation. "
        "Powered by LangGraph + Gemini."
    ),
    version="0.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS Hardening ────────────────────────────────────────────────────────────
cors_origins = list(settings.cors_allowed_origins)
if settings.is_development and not cors_origins:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from backend.routers.auth import router as auth_router  # noqa: E402
from backend.routers.agent import router as agent_router  # noqa: E402
from backend.routers.documents import router as documents_router  # noqa: E402
from backend.routers.students import router as students_router  # noqa: E402
from backend.routers.placement import router as placement_router  # noqa: E402
from backend.routers.interview import interview_router  # noqa: E402
app.include_router(auth_router)
app.include_router(agent_router)
app.include_router(documents_router)
app.include_router(students_router)
app.include_router(placement_router)
app.include_router(interview_router, prefix="/api/interview", tags=["Mock Interview Agent"])





# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Comprehensive readiness and liveness probe checking DB, ChromaDB, and LLM configuration."""
    from sqlalchemy import text
    from database.connection import get_db_context

    db_status = "healthy"
    try:
        with get_db_context() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    chroma_status = "ready"
    try:
        from rag.vectorstore import get_collection_stats
        c_stats = get_collection_stats()
        chroma_status = c_stats.get("status", "ready")
    except Exception as exc:
        chroma_status = f"degraded: {exc}"

    api_key_configured = bool(settings.gemini_api_key)
    is_ready = db_status == "healthy" and (api_key_configured or settings.is_development)

    return {
        "status": "ok" if is_ready else "degraded",
        "app": "CareerPilot",
        "version": "0.3.0",
        "environment": settings.app_env,
        "services": {
            "database": db_status,
            "chromadb": chroma_status,
            "gemini_api_key": "configured" if api_key_configured else "missing",
        },
    }


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root() -> dict:
    return {
        "message": "Welcome to CareerPilot API",
        "docs": "/docs",
        "health": "/health",
    }


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
