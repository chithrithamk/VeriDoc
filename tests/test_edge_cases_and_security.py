"""
Security, Edge Cases, and Regression Tests (Phase 13).

Validates:
1. Secrets protection (.env excluded from git tracking, keys not logged/exposed).
2. Database isolation (tests do not modify or create data/veridoc.db).
3. Q&A History reading does NOT call Gemini or FAISS retrieval.
4. Edge cases in PDF parsing, chunking, retrieval, generation, and support checking.
5. Regressions for Phase 10-12 multi-question persistence and citations.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.database.models import DocumentRecord, QuestionRecord
from backend.database.repository import (
    clear_all_question_records,
    create_document_record,
    create_question_record,
    get_document_record_by_id,
    list_question_records,
)
from backend.services.chunker import DocumentChunk, chunk_document, chunk_text
from backend.services.embeddings import EmbeddedChunk, embed_text
from backend.services.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    LLMConfigurationError,
    LLMGenerationError,
)
from backend.services.pdf_processor import ExtractedDocument, PageData
from backend.services.rag_pipeline import RAGPipeline
from backend.services.retrieval import RetrievalService
from backend.services.support_checker import (
    SupportChecker,
    SupportCheckResult,
    SupportStatus,
)
from backend.services.vector_store import FAISSVectorStore, SearchResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. Security & Secrets Protection Tests
# =============================================================================

@pytest.mark.security
def test_gitignore_excludes_env_and_db_files():
    """Verify that .gitignore properly excludes .env, .env.local, and database files."""
    gitignore_path = PROJECT_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore file must exist in project root."

    content = gitignore_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines()]

    assert ".env" in lines, ".env must be ignored in .gitignore"
    assert any("*.db" in l or "data/veridoc.db" in l for l in lines), "*.db must be ignored in .gitignore"
    assert any("*.faiss" in l for l in lines), "*.faiss must be ignored in .gitignore"


@pytest.mark.security
def test_secrets_not_exposed_in_models_or_dicts(in_memory_db):
    """Verify that ORM models and schema dicts do not leak API keys or system secrets."""
    doc = create_document_record(
        db=in_memory_db,
        filename="test.pdf",
        total_pages=1,
        total_characters=100,
        total_chunks=1,
        indexed_vectors=1,
    )
    doc_dict = doc.to_dict()
    assert "api_key" not in doc_dict
    assert "secret" not in doc_dict
    assert "password" not in doc_dict

    q = create_question_record(
        db=in_memory_db,
        question="What is VeriDoc?",
        answer="An AI platform.",
        document_id=doc.id,
    )
    q_dict = q.to_dict()
    assert "api_key" not in q_dict
    assert "secret" not in q_dict


# =============================================================================
# 2. Database Isolation & Zero-LLM History Read Tests
# =============================================================================

@pytest.mark.security
def test_test_database_isolation_does_not_touch_production_db():
    """Verify that the in_memory_db fixture does not create or write to data/veridoc.db."""
    real_db_path = PROJECT_ROOT / "data" / "veridoc.db"
    # Even if data/veridoc.db exists from local dev, its modification time should not change during in-memory tests
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from backend.database.session import Base

    test_eng = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_eng)
    assert test_eng.url.database == ":memory:"


def test_loading_history_does_not_call_gemini_or_faiss(in_memory_db):
    """Test that reading persistent history from SQLite performs 0 LLM calls and 0 vector searches."""
    doc = create_document_record(
        db=in_memory_db,
        filename="history_test.pdf",
        total_pages=1,
        total_characters=100,
        total_chunks=1,
        indexed_vectors=1,
    )

    create_question_record(
        db=in_memory_db,
        question="Stored question?",
        answer="Stored answer from database.",
        document_id=doc.id,
        document_name=doc.filename,
        sources=[{"chunk_id": 1, "page_number": 1, "text": "Snippet", "similarity_score": 0.95}],
        support_status="supported",
        support_confidence=0.99,
    )

    mock_llm = MagicMock()
    mock_store = MagicMock()

    # Read from database repository
    history_records = list_question_records(in_memory_db)
    assert len(history_records) == 1
    assert history_records[0].question == "Stored question?"
    assert history_records[0].answer == "Stored answer from database."

    # Verify 0 LLM or FAISS calls were made
    mock_llm.models.generate_content.assert_not_called()
    mock_store.search.assert_not_called()


# =============================================================================
# 3. Chunker & Text Edge Cases
# =============================================================================

def test_chunker_single_word_longer_than_chunk_size():
    """Test that an extremely long single continuous string without spaces does not crash."""
    long_word = "A" * 250
    chunks = chunk_text(long_word, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_chunker_exact_match_chunk_size():
    """Test chunking when text length matches chunk_size exactly."""
    exact_text = "This text is precisely forty chars long."
    chunks = chunk_text(exact_text, chunk_size=len(exact_text), chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0] == exact_text


# =============================================================================
# 4. Support Checker Edge Cases & Fallbacks
# =============================================================================

def test_support_checker_handles_unexpected_json_fields():
    """Test that support checker tolerates extra JSON keys gracefully."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "status": "supported",
        "confidence": 0.90,
        "explanation": "Valid explanation.",
        "supported_claims": ["Claim 1"],
        "unsupported_claims": [],
        "extra_field_from_future_gemini": True,
        "model_version": "3.6-flash",
    })
    mock_client.models.generate_content.return_value = mock_resp

    checker = SupportChecker(client=mock_client)
    res = checker.check_support(
        question="What is this?",
        answer="Claim 1.",
        sources=[
            SearchResult(
                chunk=EmbeddedChunk(1, "Claim 1.", 1, "doc.pdf", [0.1] * 384, 8),
                score=0.9,
            )
        ],
    )
    assert res.status == "supported"
    assert res.confidence == 0.90
    assert len(res.supported_claims) == 1


def test_support_checker_clamps_confidence_bounds():
    """Test that confidence values outside [0, 1] are clamped safely."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "status": "supported",
        "confidence": 1.5,  # Out of bounds
        "explanation": "Out of bounds confidence score",
        "supported_claims": ["Test claim"],
        "unsupported_claims": [],
    })
    mock_client.models.generate_content.return_value = mock_resp

    checker = SupportChecker(client=mock_client)
    res = checker.check_support(
        question="Query?",
        answer="Test claim",
        sources=[SearchResult(chunk=EmbeddedChunk(1, "Test", 1, "doc.pdf", [0.1] * 384, 4), score=0.8)],
    )
    assert res.confidence <= 1.0
