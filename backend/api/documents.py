"""
VeriDoc — Document Ingestion and Management API Router (Phase 8)
"""

from pathlib import Path
import shutil
import tempfile
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from backend.models.schemas import DocumentStatsResponse, DocumentUploadResponse
from backend.services.pdf_processor import (
    CorruptedPDFError,
    InvalidPDFError,
    PDFNotFoundError,
    PDFProcessingError,
)
from backend.services.rag_pipeline import RAGPipeline

router = APIRouter(tags=["documents"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_pipeline() -> RAGPipeline:
    """Dependency provider returning the active RAGPipeline instance."""
    from backend.main import get_rag_pipeline
    return get_rag_pipeline()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF document",
    description="Uploads a PDF document, extracts text page-by-page, chunks text, generates embeddings, and builds a FAISS vector index.",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document file to upload and index"),
    chunk_size: int = Query(default=1000, ge=100, le=5000, description="Target character size per chunk"),
    chunk_overlap: int = Query(default=200, ge=0, le=1000, description="Character overlap between chunks"),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentUploadResponse:
    """Handles PDF file upload, text extraction, chunking, and FAISS indexing."""
    filename = file.filename or "uploaded_document.pdf"

    # Validate file extension
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file format for '{filename}'. Only .pdf files are supported.",
        )

    # Validate overlap constraint
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be strictly less than chunk_size.",
        )

    # Save uploaded file temporarily for PyMuPDF processing
    temp_dir = PROJECT_ROOT / "data" / "documents"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_dir / f"api_temp_{uuid.uuid4().hex[:8]}_{filename}"

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest document into the RAG pipeline
        stats = pipeline.ingest_pdf(
            pdf_path=temp_file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        return DocumentUploadResponse(
            filename=filename,
            total_pages=stats["total_pages"],
            total_characters=stats["total_characters"],
            total_chunks=stats["total_chunks"],
            indexed_vectors=stats["indexed_vectors"],
            status="success",
            message=f"Document '{filename}' successfully processed and indexed ({stats['total_chunks']} chunks).",
        )

    except (InvalidPDFError, CorruptedPDFError, PDFNotFoundError, ValueError) as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except PDFProcessingError as err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process PDF: {err}",
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during document ingestion: {err}",
        )
    finally:
        # Clean up temporary upload file
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass


@router.get(
    "/stats",
    response_model=DocumentStatsResponse,
    summary="Get current document and index statistics",
    description="Returns metadata about the currently loaded and indexed PDF document.",
)
async def get_document_stats(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> DocumentStatsResponse:
    """Returns metadata for the currently active document and vector store."""
    stats = pipeline.get_stats()
    return DocumentStatsResponse(
        document_name=stats["document_name"],
        total_pages=stats["total_pages"],
        total_characters=stats["total_characters"],
        total_chunks=stats["total_chunks"],
        indexed_vectors=stats["indexed_vectors"],
        is_ready=stats["is_ready"],
    )


@router.delete(
    "/clear",
    summary="Clear active document and index",
    description="Resets the in-memory document state and clears the FAISS vector index.",
)
async def clear_document(
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> dict:
    """Clears the currently loaded document and vector index."""
    pipeline.clear()
    return {"status": "success", "message": "Document index cleared successfully."}
