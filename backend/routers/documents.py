"""
CareerPilot — Documents & RAG API Router (Phase 4)

Exposes:
  - POST /api/documents/ingest   — Ingest & chunk files from data/documents/ into ChromaDB
  - GET  /api/documents/status   — Query current ChromaDB index status & document list
  - POST /api/documents/query    — Semantic similarity retrieval endpoint
  - POST /api/documents/explain  — Generate interview explanation grounded in retrieved material
  - POST /api/documents/reset    — Clear vector store collection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from agent.llm import get_llm
from agent.prompts import EXPLANATION_PROMPT
from backend.config import get_settings
from rag.ingestion import SUPPORTED_DOCUMENT_EXTENSIONS, load_and_chunk_directory
from rag.retriever import retrieve_context_string, retrieve_relevant_documents
from rag.vectorstore import get_collection_stats, index_documents, reset_collection

router = APIRouter(prefix="/api/documents", tags=["RAG & Documents"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks_indexed: int
    total_in_collection: int
    files_discovered: List[str]


class DocumentChunkResult(BaseModel):
    content: str
    source: str
    chunk_index: int


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        examples=["Java HashMap collision resolution"],
        description="Search query to retrieve study materials for.",
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    total_retrieved: int
    chunks: List[DocumentChunkResult]
    context_text: str


class ExplainRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        examples=["Explain the difference between HashMap and ConcurrentHashMap in Java."],
        description="Interview concept or question to explain using study materials.",
    )


class ExplainResponse(BaseModel):
    query: str
    explanation: str
    grounded_in_documents: bool
    context_used: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into ChromaDB",
    description="Scans data/documents/, chunks all supported files, generates embeddings, and stores them in ChromaDB.",
)
async def ingest_documents() -> IngestResponse:
    """Ingest all documents found in the configured documents directory."""
    settings = get_settings()
    docs_dir = settings.documents_dir

    logger.info(f"Triggered document ingestion from: {docs_dir}")

    # Discover files
    files = [
        f.name for f in docs_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]

    try:
        chunks = load_and_chunk_directory(docs_dir)
        num_indexed = index_documents(chunks)
        stats = get_collection_stats()

        return IngestResponse(
            success=True,
            message=f"Successfully ingested and indexed {num_indexed} chunks from {len(files)} files.",
            chunks_indexed=num_indexed,
            total_in_collection=stats.get("total_chunks", 0),
            files_discovered=files,
        )
    except Exception as exc:
        logger.error(f"Ingestion failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(exc)}",
        )


@router.get(
    "/status",
    summary="Get ChromaDB collection status",
    description="Returns current document and chunk counts in ChromaDB.",
)
async def get_status() -> Dict[str, Any]:
    """Return index status and list of files currently on disk."""
    settings = get_settings()
    stats = get_collection_stats()

    files = [
        f.name for f in settings.documents_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ] if settings.documents_dir.exists() else []

    return {
        "status": stats.get("status", "ready"),
        "total_chunks_in_chroma": stats.get("total_chunks", 0),
        "collection_name": stats.get("collection_name"),
        "documents_on_disk": files,
        "documents_directory": str(settings.documents_dir),
    }


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query study materials semantically",
    description="Searches ChromaDB vector store for chunks matching the query.",
)
async def query_documents(request: QueryRequest) -> QueryResponse:
    """Perform similarity search over indexed study materials."""
    try:
        raw_docs = retrieve_relevant_documents(request.query, top_k=request.top_k)
        context_str = retrieve_context_string(request.query, top_k=request.top_k)

        chunk_results = [
            DocumentChunkResult(
                content=d.page_content,
                source=d.metadata.get("source", "Unknown"),
                chunk_index=d.metadata.get("chunk_index", 0),
            )
            for d in raw_docs
        ]

        return QueryResponse(
            query=request.query,
            total_retrieved=len(chunk_results),
            chunks=chunk_results,
            context_text=context_str,
        )
    except Exception as exc:
        logger.error(f"Document query error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval failed: {str(exc)}",
        )


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="Generate RAG-grounded topic explanation",
    description="Retrieves relevant study material from ChromaDB and uses Gemini to generate a placement-oriented explanation.",
)
async def explain_topic(request: ExplainRequest) -> ExplainResponse:
    """Answer an interview question or explain a concept grounded in study materials."""
    context = retrieve_context_string(request.query, top_k=3)
    has_context = bool(context.strip())

    try:
        llm = get_llm()
        chain = EXPLANATION_PROMPT | llm
        response = chain.invoke({
            "query": request.query,
            "context": context if has_context else "No custom documents found. Use standard industry knowledge.",
        })
        raw_content = response.content if hasattr(response, "content") else str(response)
        if isinstance(raw_content, list):
            parts = []
            for item in raw_content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                elif hasattr(item, "text"):
                    parts.append(getattr(item, "text"))
                else:
                    parts.append(str(item))
            explanation_text = "\n".join(parts)
        else:
            explanation_text = str(raw_content)

        return ExplainResponse(
            query=request.query,
            explanation=explanation_text.strip(),
            grounded_in_documents=has_context,
            context_used=context,
        )
    except Exception as exc:
        logger.error(f"Explanation generation error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(exc)}",
        )


@router.post(
    "/reset",
    summary="Reset vector store",
    description="Clears all indexed chunks from the ChromaDB collection.",
)
async def reset_documents() -> Dict[str, Any]:
    """Clear ChromaDB collection."""
    success = reset_collection()
    return {"success": success, "message": "ChromaDB collection cleared."}
