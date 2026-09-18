"""
VeriDoc — Document Ingestion and Persistent Database API Router (Phases 8 & 9)
"""

from pathlib import Path
import shutil
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from backend.database.repository import (
    clear_all_document_records,
    count_question_records,
    create_document_record,
    get_document_record_by_id,
    get_latest_document_record,
    list_document_records,
    list_questions_for_document,
)
from backend.database.session import get_db
from backend.models.schemas import (
    DocumentListResponse,
    DocumentRecordResponse,
    DocumentStatsResponse,
    DocumentUploadResponse,
    QuestionHistoryItemResponse,
    QuestionHistoryListResponse,
    SourceCitation,
    SupportCheckResponse,
)
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
    description="Uploads a PDF document, extracts text page-by-page, chunks text, generates embeddings, builds a FAISS vector index, and stores persistent metadata in SQLite.",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document file to upload and index"),
    chunk_size: int = Query(default=400, ge=100, le=5000, description="Target character size per chunk"),
    chunk_overlap: int = Query(default=80, ge=0, le=1000, description="Character overlap between chunks"),
    pipeline: RAGPipeline = Depends(get_pipeline),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    """Handles PDF file upload, text extraction, chunking, FAISS indexing, and database persistence."""
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

    # Save uploaded file temporarily in an isolated folder to preserve original filename
    temp_subdir = PROJECT_ROOT / "data" / "documents" / f"upload_{uuid.uuid4().hex[:8]}"
    temp_subdir.mkdir(parents=True, exist_ok=True)
    temp_file_path = temp_subdir / filename

    file_size_bytes = 0
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size_bytes = temp_file_path.stat().st_size

        # Ingest document into the RAG pipeline
        stats = pipeline.ingest_pdf(
            pdf_path=temp_file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Persist structured document metadata into SQLite
        record = create_document_record(
            db=db,
            filename=filename,
            total_pages=stats["total_pages"],
            total_characters=stats["total_characters"],
            total_chunks=stats["total_chunks"],
            indexed_vectors=stats["indexed_vectors"],
            file_size_bytes=file_size_bytes,
            status="indexed",
        )

        return DocumentUploadResponse(
            id=record.id,
            filename=filename,
            total_pages=stats["total_pages"],
            total_characters=stats["total_characters"],
            total_chunks=stats["total_chunks"],
            indexed_vectors=stats["indexed_vectors"],
            status="success",
            message=f"Document '{filename}' successfully processed, indexed in FAISS, and saved to database.",
            created_at=record.created_at.isoformat() if record.created_at else None,
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
        # Clean up temporary upload file and folder
        if temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass
        if temp_subdir.exists():
            try:
                temp_subdir.rmdir()
            except Exception:
                pass


@router.get(
    "/stats",
    response_model=DocumentStatsResponse,
    summary="Get current document and index statistics",
    description="Returns metadata about the currently loaded and indexed PDF document from memory and database.",
)
async def get_document_stats(
    pipeline: RAGPipeline = Depends(get_pipeline),
    db: Session = Depends(get_db),
) -> DocumentStatsResponse:
    """Returns metadata for the currently active document and vector store."""
    stats = pipeline.get_stats()
    latest_record = get_latest_document_record(db)

    return DocumentStatsResponse(
        id=latest_record.id if (latest_record and stats["is_ready"]) else None,
        document_name=stats["document_name"],
        total_pages=stats["total_pages"],
        total_characters=stats["total_characters"],
        total_chunks=stats["total_chunks"],
        indexed_vectors=stats["indexed_vectors"],
        is_ready=stats["is_ready"],
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all persistent document records",
    description="Retrieves a list of all ingested document records stored in SQLite.",
)
async def list_documents(
    limit: int = Query(default=50, ge=1, le=100, description="Max documents to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """Lists all persistent document records."""
    records = list_document_records(db=db, limit=limit, offset=offset)
    total_count = len(records)
    doc_responses = [
        DocumentRecordResponse(
            id=r.id,
            filename=r.filename,
            file_path=r.file_path,
            file_size_bytes=r.file_size_bytes,
            total_pages=r.total_pages,
            total_characters=r.total_characters,
            total_chunks=r.total_chunks,
            indexed_vectors=r.indexed_vectors,
            status=r.status,
            created_at=r.created_at.isoformat(),
            updated_at=r.updated_at.isoformat(),
        )
        for r in records
    ]
    return DocumentListResponse(total=total_count, documents=doc_responses)


@router.get(
    "/{doc_id}",
    response_model=DocumentRecordResponse,
    summary="Get persistent document record by ID",
    description="Retrieves metadata for a specific document from SQLite.",
)
async def get_document_by_id(
    doc_id: str,
    db: Session = Depends(get_db),
) -> DocumentRecordResponse:
    """Retrieves a single persistent document record by ID."""
    record = get_document_record_by_id(db=db, doc_id=doc_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return DocumentRecordResponse(
        id=record.id,
        filename=record.filename,
        file_path=record.file_path,
        file_size_bytes=record.file_size_bytes,
        total_pages=record.total_pages,
        total_characters=record.total_characters,
        total_chunks=record.total_chunks,
        indexed_vectors=record.indexed_vectors,
        status=record.status,
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
    )


@router.get(
    "/{doc_id}/questions",
    response_model=QuestionHistoryListResponse,
    summary="Get persistent Q&A history for a specific document",
    description="Retrieves all question records associated with a document ID from SQLite.",
)
async def get_document_questions(
    doc_id: str,
    limit: int = Query(default=50, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
) -> QuestionHistoryListResponse:
    """Retrieves question history for a given document ID."""
    doc = get_document_record_by_id(db=db, doc_id=doc_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )

    records = list_questions_for_document(db=db, document_id=doc_id, limit=limit, offset=offset)
    total = count_question_records(db=db, document_id=doc_id)

    items = []
    for r in records:
        sources = [
            SourceCitation(
                chunk_id=s.get("chunk_id", 0),
                page_number=s.get("page_number", 1),
                document_name=s.get("document_name", "document.pdf"),
                char_count=s.get("char_count", len(s.get("text", ""))),
                text=s.get("text", ""),
                similarity_score=float(s.get("similarity_score", 0.0)),
            )
            for s in (r.sources_data or [])
        ]
        supp = None
        if r.support_status:
            supp = SupportCheckResponse(
                status=r.support_status,
                confidence=float(r.support_confidence) if r.support_confidence is not None else 0.0,
                explanation=r.support_explanation or "",
                supported_claims=list(r.supported_claims) if r.supported_claims else [],
                unsupported_claims=list(r.unsupported_claims) if r.unsupported_claims else [],
            )
        items.append(
            QuestionHistoryItemResponse(
                id=r.id,
                document_id=r.document_id,
                document_name=r.document_name,
                question=r.question,
                answer=r.answer,
                sources=sources,
                support=supp,
                created_at=r.created_at.isoformat(),
            )
        )

    return QuestionHistoryListResponse(total=total, questions=items)


@router.delete(
    "/clear",
    summary="Clear active document and index",
    description="Resets the in-memory document state, clears the FAISS vector index, and optionally clears DB records.",
)
async def clear_document(
    pipeline: RAGPipeline = Depends(get_pipeline),
    db: Session = Depends(get_db),
) -> dict:
    """Clears the currently loaded document and vector index."""
    pipeline.clear()
    return {"status": "success", "message": "Document index cleared successfully."}
