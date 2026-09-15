"""
VeriDoc — Question Answering, Retrieval, and Persistent History API Router (Phases 8, 11 & 12)
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database.repository import (
    clear_all_question_records,
    count_question_records,
    create_question_record,
    delete_question_record,
    get_latest_document_record,
    get_question_record_by_id,
    list_question_records,
    list_questions_for_document,
)
from backend.database.session import get_db
from backend.models.schemas import (
    QuestionHistoryItemResponse,
    QuestionHistoryListResponse,
    QuestionRequest,
    QuestionResponse,
    SourceCitation,
    SupportCheckResponse,
)
from backend.services.generator import (
    GeneratedAnswer,
    LLMConfigurationError,
    LLMGenerationError,
)
from backend.services.rag_pipeline import RAGPipeline

router = APIRouter(tags=["questions"])


def get_pipeline() -> RAGPipeline:
    """Dependency provider returning the active RAGPipeline instance."""
    from backend.main import get_rag_pipeline
    return get_rag_pipeline()


def _format_question_history_item(record) -> QuestionHistoryItemResponse:
    """Helper to convert a QuestionRecord ORM instance to a QuestionHistoryItemResponse schema."""
    sources = [
        SourceCitation(
            chunk_id=s.get("chunk_id", 0),
            page_number=s.get("page_number", 1),
            document_name=s.get("document_name", "document.pdf"),
            char_count=s.get("char_count", len(s.get("text", ""))),
            text=s.get("text", ""),
            similarity_score=float(s.get("similarity_score", 0.0)),
        )
        for s in (record.sources_data or [])
    ]

    support_response = None
    if record.support_status:
        support_response = SupportCheckResponse(
            status=record.support_status,
            confidence=float(record.support_confidence) if record.support_confidence is not None else 0.0,
            explanation=record.support_explanation or "",
            supported_claims=list(record.supported_claims) if record.supported_claims else [],
            unsupported_claims=list(record.unsupported_claims) if record.unsupported_claims else [],
        )

    return QuestionHistoryItemResponse(
        id=record.id,
        document_id=record.document_id,
        document_name=record.document_name,
        question=record.question,
        answer=record.answer,
        sources=sources,
        support=support_response,
        created_at=record.created_at.isoformat(),
    )


# -----------------------------------------------------------------------------
# Q&A Execution & Persistence
# -----------------------------------------------------------------------------

@router.post(
    "/ask",
    response_model=QuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against the indexed document",
    description="Performs semantic retrieval over FAISS, generates a grounded response, checks factual support, and persists the interaction to SQLite.",
)
async def ask_question(
    request: QuestionRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
    db: Session = Depends(get_db),
) -> QuestionResponse:
    """Retrieves relevant chunks, generates a grounded answer, verifies support, and stores Q&A history."""
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace-only.",
        )

    if not pipeline.is_ready():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No document has been processed or indexed yet. Upload a PDF document first.",
        )

    try:
        top_k = request.top_k or 5
        generated_answer: GeneratedAnswer = pipeline.ask(
            question=clean_question,
            top_k=top_k,
            check_support=True,
        )

        citations = [
            SourceCitation(
                chunk_id=src.chunk.chunk_id,
                page_number=src.chunk.page_number,
                document_name=src.chunk.document_name,
                char_count=src.chunk.char_count,
                text=src.chunk.text,
                similarity_score=src.score,
            )
            for src in generated_answer.sources
        ]

        support_response = None
        if generated_answer.support is not None:
            supp = generated_answer.support
            support_response = SupportCheckResponse(
                status=supp.status,
                confidence=float(supp.confidence),
                explanation=supp.explanation,
                supported_claims=list(supp.supported_claims),
                unsupported_claims=list(supp.unsupported_claims),
            )

        # Retrieve active document record from database
        active_doc = get_latest_document_record(db)
        doc_id = active_doc.id if active_doc else None
        doc_name = active_doc.filename if active_doc else (
            citations[0].document_name if citations else "uploaded_document.pdf"
        )

        # Persist Q&A record in SQLite
        sources_dicts = [c.model_dump() for c in citations]
        q_record = create_question_record(
            db=db,
            question=clean_question,
            answer=generated_answer.answer,
            document_id=doc_id,
            document_name=doc_name,
            sources=sources_dicts,
            support_status=support_response.status if support_response else None,
            support_confidence=support_response.confidence if support_response else None,
            support_explanation=support_response.explanation if support_response else None,
            supported_claims=support_response.supported_claims if support_response else [],
            unsupported_claims=support_response.unsupported_claims if support_response else [],
        )

        return QuestionResponse(
            id=q_record.id,
            document_id=q_record.document_id,
            document_name=q_record.document_name,
            question=clean_question,
            answer=generated_answer.answer,
            sources=citations,
            support=support_response,
            created_at=q_record.created_at.isoformat() if q_record.created_at else None,
        )

    except LLMConfigurationError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM Configuration Error: {err}",
        )
    except LLMGenerationError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM Generation Error: {err}",
        )
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except RuntimeError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during question answering: {err}",
        )


# -----------------------------------------------------------------------------
# Persistent Q&A History Endpoints
# -----------------------------------------------------------------------------

@router.get(
    "/history",
    response_model=QuestionHistoryListResponse,
    summary="List persistent Q&A history records",
    description="Retrieves persistent questions and answers across documents with pagination.",
)
async def get_question_history(
    limit: int = Query(default=50, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    document_id: Optional[str] = Query(default=None, description="Optional document ID filter"),
    db: Session = Depends(get_db),
) -> QuestionHistoryListResponse:
    """Lists question history records."""
    if document_id:
        records = list_questions_for_document(db=db, document_id=document_id, limit=limit, offset=offset)
        total = count_question_records(db=db, document_id=document_id)
    else:
        records = list_question_records(db=db, limit=limit, offset=offset)
        total = count_question_records(db=db)

    items = [_format_question_history_item(r) for r in records]
    return QuestionHistoryListResponse(total=total, questions=items)


@router.get(
    "/history/{question_id}",
    response_model=QuestionHistoryItemResponse,
    summary="Get persistent question record by ID",
    description="Retrieves a specific Q&A history item by UUID.",
)
async def get_question_by_id(
    question_id: str,
    db: Session = Depends(get_db),
) -> QuestionHistoryItemResponse:
    """Retrieves a single question history record."""
    record = get_question_record_by_id(db=db, question_id=question_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question record with ID '{question_id}' not found.",
        )
    return _format_question_history_item(record)


@router.delete(
    "/history/{question_id}",
    summary="Delete a question history record",
    description="Deletes a specific Q&A history item by UUID.",
)
async def delete_question(
    question_id: str,
    db: Session = Depends(get_db),
) -> dict:
    """Deletes a question history record by ID."""
    deleted = delete_question_record(db=db, question_id=question_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Question record with ID '{question_id}' not found.",
        )
    return {"status": "success", "message": f"Question '{question_id}' deleted successfully."}


@router.delete(
    "/history",
    summary="Clear all or document-specific Q&A history",
    description="Clears Q&A history from SQLite without deleting document records.",
)
async def clear_question_history(
    document_id: Optional[str] = Query(default=None, description="Optional document ID to clear history for"),
    db: Session = Depends(get_db),
) -> dict:
    """Clears question history records."""
    count = clear_all_question_records(db=db, document_id=document_id)
    msg = f"Cleared {count} question history records for document '{document_id}'." if document_id else f"Cleared {count} question history records."
    return {"status": "success", "deleted_count": count, "message": msg}
