"""
Unit and integration tests for Phase 11: Hallucination & Factual Support Checking.
"""

import io
import json
from unittest.mock import MagicMock, patch
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.session import Base, get_db
from backend.main import app, set_rag_pipeline
from backend.models.schemas import QuestionResponse, SupportCheckResponse
from backend.services.chunker import DocumentChunk
from backend.services.embeddings import EmbeddedChunk
from backend.services.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    LLMConfigurationError,
)
from backend.services.pdf_processor import ExtractedDocument, PageData
from backend.services.rag_pipeline import RAGPipeline
from backend.services.support_checker import (
    SupportChecker,
    SupportCheckResult,
    SupportStatus,
)
from backend.services.vector_store import SearchResult


@pytest.fixture
def sample_sources():
    """Create sample SearchResult objects representing retrieved evidence."""
    chunk1 = EmbeddedChunk(
        chunk_id=1,
        text="VeriDoc uses PyMuPDF for high-speed page-level PDF text extraction.",
        page_number=1,
        document_name="doc.pdf",
        char_count=68,
        embedding=[0.1] * 384,
    )
    chunk2 = EmbeddedChunk(
        chunk_id=2,
        text="Dense vector similarity search is performed using FAISS IndexFlatIP.",
        page_number=2,
        document_name="doc.pdf",
        char_count=69,
        embedding=[0.2] * 384,
    )
    return [
        SearchResult(chunk=chunk1, score=0.91),
        SearchResult(chunk=chunk2, score=0.88),
    ]


@pytest.fixture
def multi_page_extracted_doc():
    """Create sample extracted document."""
    return ExtractedDocument(
        file_path="sample.pdf",
        total_pages=2,
        pages=[
            PageData(page_number=1, text="VeriDoc uses PyMuPDF for text extraction.", char_count=42),
            PageData(page_number=2, text="FAISS handles fast similarity indexing.", char_count=39),
        ],
        total_characters=81,
        has_text=True,
    )


# -----------------------------------------------------------------------------
# 1. Unit Tests for SupportChecker
# -----------------------------------------------------------------------------

def test_support_checker_initialization_missing_api_key():
    """Test that missing API key raises LLMConfigurationError."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLMConfigurationError, match="Gemini API key is not configured"):
            SupportChecker(api_key=None, client=None)


def test_fully_supported_answer(sample_sources):
    """Test that a fully supported answer returns status='supported'."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "supported",
        "confidence": 0.95,
        "explanation": "All claims are directly corroborated by the document passages.",
        "supported_claims": [
            "VeriDoc uses PyMuPDF for text extraction.",
            "FAISS is used for vector search."
        ],
        "unsupported_claims": []
    })
    mock_client.models.generate_content.return_value = mock_response

    checker = SupportChecker(client=mock_client)
    res = checker.check_support(
        question="What libraries does VeriDoc use?",
        answer="VeriDoc uses PyMuPDF for text extraction and FAISS for vector search.",
        sources=sample_sources,
    )

    assert res.status == SupportStatus.SUPPORTED.value
    assert res.confidence == 0.95
    assert len(res.supported_claims) == 2
    assert len(res.unsupported_claims) == 0
    assert "directly corroborated" in res.explanation


def test_partially_supported_answer(sample_sources):
    """Test that an answer with mixed claims returns status='partially_supported'."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "partially_supported",
        "confidence": 0.85,
        "explanation": "PyMuPDF is supported, but Redis is not mentioned in the text.",
        "supported_claims": ["VeriDoc uses PyMuPDF for PDF text extraction."],
        "unsupported_claims": ["VeriDoc uses Redis for caching."]
    })
    mock_client.models.generate_content.return_value = mock_response

    checker = SupportChecker(client=mock_client)
    res = checker.check_support(
        question="What does VeriDoc use?",
        answer="VeriDoc uses PyMuPDF for PDF extraction and Redis for caching.",
        sources=sample_sources,
    )

    assert res.status == SupportStatus.PARTIALLY_SUPPORTED.value
    assert res.confidence == 0.85
    assert len(res.supported_claims) == 1
    assert len(res.unsupported_claims) == 1
    assert "Redis" in res.unsupported_claims[0]


def test_unsupported_hallucinated_answer(sample_sources):
    """Test that an unsupported hallucinated answer returns status='unsupported'."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps({
        "status": "unsupported",
        "confidence": 0.92,
        "explanation": "None of the claims match the retrieved document context.",
        "supported_claims": [],
        "unsupported_claims": ["The document describes quantum physics algorithms."]
    })
    mock_client.models.generate_content.return_value = mock_response

    checker = SupportChecker(client=mock_client)
    res = checker.check_support(
        question="What is this document about?",
        answer="The document describes quantum physics algorithms.",
        sources=sample_sources,
    )

    assert res.status == SupportStatus.UNSUPPORTED.value
    assert res.confidence == 0.92
    assert len(res.unsupported_claims) == 1


def test_empty_sources_returns_insufficient_evidence():
    """Test that passing empty sources immediately returns status='insufficient_evidence' without calling LLM."""
    mock_client = MagicMock()
    checker = SupportChecker(client=mock_client)

    res = checker.check_support(
        question="What is VeriDoc?",
        answer="VeriDoc is an AI document assistant.",
        sources=[],
    )

    assert res.status == SupportStatus.INSUFFICIENT_EVIDENCE.value
    assert res.confidence == 1.0
    mock_client.models.generate_content.assert_not_called()


def test_refusal_answer_returns_insufficient_evidence(sample_sources):
    """Test that standard refusal answer returns status='insufficient_evidence'."""
    mock_client = MagicMock()
    checker = SupportChecker(client=mock_client)

    res = checker.check_support(
        question="What is the budget?",
        answer="The answer cannot be determined from the provided document context.",
        sources=sample_sources,
    )

    assert res.status == SupportStatus.INSUFFICIENT_EVIDENCE.value
    assert res.confidence == 1.0
    mock_client.models.generate_content.assert_not_called()


def test_malformed_json_response_handled_safely(sample_sources):
    """Test that malformed/non-JSON LLM output is caught and returns verification_unavailable."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is not valid JSON at all!"
    mock_client.models.generate_content.return_value = mock_response

    checker = SupportChecker(client=mock_client)
    res = checker.check_support(
        question="Test query",
        answer="Test answer",
        sources=sample_sources,
    )

    assert res.status == SupportStatus.VERIFICATION_UNAVAILABLE.value
    assert res.confidence == 0.0
    assert "could not be parsed" in res.explanation


def test_markdown_code_block_json_parsed_correctly(sample_sources):
    """Test that JSON wrapped in markdown code blocks ```json ... ``` is parsed cleanly."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "```json\n{\n  \"status\": \"supported\",\n  \"confidence\": 0.98,\n  \"explanation\": \"Grounded.\",\n  \"supported_claims\": [\"claim1\"],\n  \"unsupported_claims\": []\n}\n```"
    mock_client.models.generate_content.return_value = mock_response

    checker = SupportChecker(client=mock_client)
    res = checker.check_support(
        question="Test query",
        answer="Test answer",
        sources=sample_sources,
    )

    assert res.status == SupportStatus.SUPPORTED.value
    assert res.confidence == 0.98
    assert res.supported_claims == ["claim1"]


def test_gemini_api_failure_returns_verification_unavailable(sample_sources):
    """Test that API exceptions (e.g. rate limit) return status='verification_unavailable' without raising."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Rate limit exceeded (429)")

    checker = SupportChecker(client=mock_client)
    res = checker.check_support(
        question="Test query",
        answer="Test answer",
        sources=sample_sources,
    )

    assert res.status == SupportStatus.VERIFICATION_UNAVAILABLE.value
    assert res.confidence == 0.0
    assert "service error" in res.explanation.lower()


# -----------------------------------------------------------------------------
# 2. Integration with RAGPipeline
# -----------------------------------------------------------------------------

def test_rag_pipeline_ask_includes_support_check(multi_page_extracted_doc):
    """Test that RAGPipeline.ask() executes support checking and attaches SupportCheckResult."""
    mock_llm_client = MagicMock()

    # Configure mock responses for generator and support_checker
    gen_resp = MagicMock()
    gen_resp.text = "VeriDoc uses PyMuPDF for text extraction."

    check_resp = MagicMock()
    check_resp.text = json.dumps({
        "status": "supported",
        "confidence": 0.96,
        "explanation": "Supported by page 1.",
        "supported_claims": ["VeriDoc uses PyMuPDF."],
        "unsupported_claims": [],
    })

    mock_llm_client.models.generate_content.side_effect = [gen_resp, check_resp]

    pipeline = RAGPipeline(api_key="mock-key", llm_client=mock_llm_client)
    pipeline.ingest_document(multi_page_extracted_doc)

    answer = pipeline.ask("What does VeriDoc use?", top_k=2, check_support=True)

    assert answer.question == "What does VeriDoc use?"
    assert answer.answer == "VeriDoc uses PyMuPDF for text extraction."
    assert answer.support is not None
    assert isinstance(answer.support, SupportCheckResult)
    assert answer.support.status == "supported"
    assert answer.support.confidence == 0.96


# -----------------------------------------------------------------------------
# 3. Integration with FastAPI Endpoints
# -----------------------------------------------------------------------------

def test_api_ask_returns_support_field():
    """Test that POST /questions/ask returns structured support verification in response."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    mock_llm = MagicMock()
    ans_resp = MagicMock()
    ans_resp.text = "VeriDoc indexes documents into FAISS."
    chk_resp = MagicMock()
    chk_resp.text = json.dumps({
        "status": "supported",
        "confidence": 0.94,
        "explanation": "Directly supported.",
        "supported_claims": ["VeriDoc indexes documents into FAISS."],
        "unsupported_claims": [],
    })
    mock_llm.models.generate_content.side_effect = [ans_resp, chk_resp]

    pipeline = RAGPipeline(api_key="mock-key", llm_client=mock_llm)
    set_rag_pipeline(pipeline)

    # Ingest sample text directly
    doc = ExtractedDocument(
        file_path="api_doc.pdf",
        total_pages=1,
        pages=[PageData(page_number=1, text="VeriDoc indexes documents into FAISS.", char_count=36)],
        total_characters=36,
        has_text=True,
    )
    pipeline.ingest_document(doc)

    client = TestClient(app)
    response = client.post("/questions/ask", json={"question": "How are documents indexed?", "top_k": 1})
    assert response.status_code == 200
    data = response.json()

    assert data["question"] == "How are documents indexed?"
    assert data["answer"] == "VeriDoc indexes documents into FAISS."
    assert "support" in data
    assert data["support"] is not None
    assert data["support"]["status"] == "supported"
    assert data["support"]["confidence"] == 0.94
    assert len(data["support"]["supported_claims"]) == 1

    set_rag_pipeline(None)
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)
