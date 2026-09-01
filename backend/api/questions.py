"""
VeriDoc — Question Answering and Retrieval API Router (Phase 8)
"""

from fastapi import APIRouter, Depends, HTTPException, status

from backend.models.schemas import QuestionRequest, QuestionResponse, SourceCitation
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


@router.post(
    "/ask",
    response_model=QuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question against the indexed document",
    description="Performs semantic retrieval over the FAISS vector index and generates a grounded response using Google Gemini.",
)
async def ask_question(
    request: QuestionRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> QuestionResponse:
    """Retrieves relevant chunks and generates a grounded answer for the user's question."""
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

        return QuestionResponse(
            question=generated_answer.question,
            answer=generated_answer.answer,
            sources=citations,
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
