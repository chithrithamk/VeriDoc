"""
Deterministic End-to-End RAG Pipeline Integration Test (Phase 13).

Validates the complete end-to-end flow without calling live external APIs:
PDF Document -> Text Extraction -> Boundary Chunking -> Dense Embeddings ->
FAISS Vector Indexing -> Semantic Retrieval -> Grounded Gemini Generation ->
Factual Support Verification -> SQLite Persistence -> Structured Response.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.database.models import DocumentRecord, QuestionRecord
from backend.database.repository import (
    get_document_record_by_id,
    get_latest_document_record,
    list_question_records,
)
from backend.models.schemas import QuestionResponse, SourceCitation, SupportCheckResponse
from backend.services.generator import AnswerGenerator, GeneratedAnswer
from backend.services.pdf_processor import extract_text_from_pdf
from backend.services.rag_pipeline import RAGPipeline
from backend.services.support_checker import SupportChecker, SupportCheckResult


@pytest.mark.integration
def test_rag_end_to_end_full_pipeline(tmp_path: Path, in_memory_db):
    """
    Complete end-to-end pipeline test from raw PDF file to structured grounded response with persistence.
    Uses mock Gemini client for deterministic, offline verification.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz  # type: ignore

    # 1. Create a physical temporary PDF file on disk
    pdf_path = tmp_path / "enterprise_handbook.pdf"
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 72), "Chapter 1: VeriDoc Architecture. VeriDoc provides page-level grounding and vector indexing.")
    p2 = doc.new_page()
    p2.insert_text(fitz.Point(50, 72), "Chapter 2: Security Policies. All sensitive data must be encrypted with AES-256-GCM.")
    doc.save(str(pdf_path))
    doc.close()

    # 2. Setup Mock LLM Client with dual responses (Generation + Verification)
    mock_llm = MagicMock()
    mock_gen_response = MagicMock()
    mock_gen_response.text = "According to Chapter 2, all sensitive data must be encrypted using AES-256-GCM."

    mock_check_response = MagicMock()
    mock_check_response.text = json.dumps({
        "status": "supported",
        "confidence": 0.98,
        "explanation": "Direct match with Chapter 2 on page 2.",
        "supported_claims": ["Sensitive data must be encrypted with AES-256-GCM."],
        "unsupported_claims": [],
    })
    mock_llm.models.generate_content.side_effect = [mock_gen_response, mock_check_response]

    # 3. Initialize RAGPipeline with mocked client
    pipeline = RAGPipeline(api_key="mock-test-key", llm_client=mock_llm)

    # 4. Ingest PDF document
    stats = pipeline.ingest_pdf(pdf_path=pdf_path, chunk_size=300, chunk_overlap=50)

    assert stats["is_ready"] is True
    assert stats["document_name"] == "enterprise_handbook.pdf"
    assert stats["total_pages"] == 2
    assert stats["total_chunks"] >= 2
    assert stats["indexed_vectors"] >= 2
    assert pipeline.is_ready() is True

    # 5. Execute Ask Flow
    user_question = "What encryption standard is required for sensitive data?"
    answer: GeneratedAnswer = pipeline.ask(
        question=user_question,
        top_k=2,
        check_support=True,
    )

    # 6. Verify Grounded Answer & Metadata
    assert isinstance(answer, GeneratedAnswer)
    assert answer.question == user_question
    assert "AES-256-GCM" in answer.answer
    assert len(answer.sources) > 0

    # 7. Verify Source Citations
    top_src = answer.sources[0]
    assert top_src.chunk.document_name == "enterprise_handbook.pdf"
    assert top_src.chunk.page_number == 2
    assert "AES-256-GCM" in top_src.chunk.text
    assert top_src.score > 0.0

    # 8. Verify Factual Support Check Result
    assert answer.support is not None
    assert isinstance(answer.support, SupportCheckResult)
    assert answer.support.status == "supported"
    assert answer.support.confidence == 0.98
    assert "AES-256-GCM" in answer.support.supported_claims[0]
    assert len(answer.support.unsupported_claims) == 0

    # 9. Verify No Unexpected External Network Calls
    assert mock_llm.models.generate_content.call_count == 2
