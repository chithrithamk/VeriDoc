"""
VeriDoc — FastAPI Backend Application Entrypoint (Phases 8 & 9)

Provides REST endpoints for:
1. Health & status inspection (/health)
2. PDF Document ingestion, FAISS indexing & SQLite persistence (/documents/upload)
3. Question answering with grounded Gemini responses & citations (/questions/ask)
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

# Ensure .env is loaded if present
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_env_file()

from backend.database.session import engine, init_db
from backend.models.schemas import HealthResponse
from backend.services.rag_pipeline import RAGPipeline

# Initialize SQLite database schema on startup
init_db()

# Global RAGPipeline singleton for FastAPI runtime
_rag_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Dependency injection provider for the shared RAGPipeline instance."""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()
    return _rag_pipeline_instance


def set_rag_pipeline(pipeline: Optional[RAGPipeline]) -> None:
    """Helper to set or reset the pipeline instance (useful in test setups)."""
    global _rag_pipeline_instance
    _rag_pipeline_instance = pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: runs startup initialization and cleanup."""
    init_db()
    yield


# Initialize FastAPI application
app = FastAPI(
    title="VeriDoc API",
    description="AI Document Intelligence Platform — Grounded RAG Backend powered by FAISS, SQLite, and Google Gemini",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# System & Health Endpoints
# -----------------------------------------------------------------------------

@app.get(
    "/",
    tags=["system"],
    summary="Root service info",
    description="Returns basic service information and running status.",
)
async def root():
    """Root endpoint."""
    return {"message": "VeriDoc API is running", "status": "healthy", "version": "0.1.0"}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
    description="Returns the health status of the API, vector index readiness, and database connection status.",
)
async def health():
    """Health check endpoint."""
    pipeline = get_rag_pipeline()

    # Verify DB connectivity
    db_status = "connected"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy",
        service="VeriDoc API",
        version="0.1.0",
        is_document_indexed=pipeline.is_ready(),
        database_status=db_status,
    )


# -----------------------------------------------------------------------------
# Register API Routers
# -----------------------------------------------------------------------------

from backend.api.documents import router as documents_router
from backend.api.questions import router as questions_router

# Direct routes for specification compatibility (/documents/upload, /questions/ask)
app.include_router(documents_router, prefix="/documents")
app.include_router(questions_router, prefix="/questions")

# Versioned API routes for production compatibility (/api/v1/documents, /api/v1/questions)
app.include_router(documents_router, prefix="/api/v1/documents", include_in_schema=False)
app.include_router(questions_router, prefix="/api/v1/questions", include_in_schema=False)
