"""
Integration and unit tests for FastAPI Backend endpoints (Phase 8).
"""

import io
from unittest.mock import MagicMock, patch
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore
import pytest
from fastapi.testclient import TestClient

from backend.main import app, set_rag_pipeline
from backend.services.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    LLMConfigurationError,
    LLMGenerationError,
)
from backend.services.rag_pipeline import RAGPipeline


@pytest.fixture
def sample_pdf_bytes():
    """Create in-memory sample PDF bytes with two pages of test text."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(fitz.Point(50, 72), "VeriDoc is an AI document platform using PyMuPDF and FAISS.")
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(50, 72), "FastAPI exposes REST endpoints for document ingestion and Q&A.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def mock_llm_client():
    """Create a mock Google Gemini client with predictable text response."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "VeriDoc uses PyMuPDF for text extraction and FAISS for vector indexing."
    client.models.generate_content.return_value = mock_response
    return client


@pytest.fixture
def test_client(mock_llm_client):
    """Create a clean TestClient with an isolated RAGPipeline instance."""
    pipeline = RAGPipeline(api_key="mock-api-key", llm_client=mock_llm_client)
    set_rag_pipeline(pipeline)
    client = TestClient(app)
    yield client
    set_rag_pipeline(None)


# -----------------------------------------------------------------------------
# System & Health Endpoint Tests
# -----------------------------------------------------------------------------

def test_root_endpoint(test_client):
    """Test that GET / returns the basic running status."""
    response = test_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "VeriDoc API" in data["message"]


def test_health_endpoint_initial_state(test_client):
    """Test that GET /health reflects unindexed initial state."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["is_document_indexed"] is False


# -----------------------------------------------------------------------------
# Document Ingestion Endpoint Tests
# -----------------------------------------------------------------------------

def test_upload_valid_pdf(test_client, sample_pdf_bytes):
    """Test uploading a valid PDF document and indexing chunks."""
    files = {
        "file": ("test_doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    response = test_client.post("/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test_doc.pdf"
    assert data["total_pages"] == 2
    assert data["total_chunks"] >= 2
    assert data["indexed_vectors"] >= 2
    assert data["status"] == "success"

    # Verify health endpoint now reflects that document is indexed
    health_res = test_client.get("/health")
    assert health_res.json()["is_document_indexed"] is True


def test_upload_invalid_file_extension(test_client):
    """Test that uploading a non-PDF file returns 400 Bad Request."""
    files = {
        "file": ("notes.txt", io.BytesIO(b"Hello text content"), "text/plain")
    }
    response = test_client.post("/documents/upload", files=files)
    assert response.status_code == 400
    assert "Only .pdf files are supported" in response.json()["detail"]


def test_upload_invalid_overlap_parameter(test_client, sample_pdf_bytes):
    """Test that chunk_overlap >= chunk_size returns 400 Bad Request."""
    files = {
        "file": ("doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    response = test_client.post("/documents/upload?chunk_size=500&chunk_overlap=500", files=files)
    assert response.status_code == 400
    assert "chunk_overlap must be strictly less than chunk_size" in response.json()["detail"]


def test_get_document_stats(test_client, sample_pdf_bytes):
    """Test GET /documents/stats returns accurate document metadata."""
    # Before upload
    res_before = test_client.get("/documents/stats")
    assert res_before.status_code == 200
    assert res_before.json()["is_ready"] is False

    # After upload
    files = {
        "file": ("stats_doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    test_client.post("/documents/upload", files=files)

    res_after = test_client.get("/documents/stats")
    assert res_after.status_code == 200
    data = res_after.json()
    assert data["is_ready"] is True
    assert data["total_pages"] == 2
    assert data["indexed_vectors"] >= 2


def test_clear_document(test_client, sample_pdf_bytes):
    """Test DELETE /documents/clear resets vector store and document state."""
    files = {
        "file": ("doc_to_clear.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    test_client.post("/documents/upload", files=files)
    assert test_client.get("/health").json()["is_document_indexed"] is True

    clear_res = test_client.delete("/documents/clear")
    assert clear_res.status_code == 200
    assert clear_res.json()["status"] == "success"

    assert test_client.get("/health").json()["is_document_indexed"] is False


# -----------------------------------------------------------------------------
# Question Answering Endpoint Tests
# -----------------------------------------------------------------------------

def test_ask_question_success(test_client, sample_pdf_bytes):
    """Test POST /questions/ask with an indexed document produces answer and citations."""
    files = {
        "file": ("qa_doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    test_client.post("/documents/upload", files=files)

    payload = {
        "question": "What is VeriDoc and what libraries does it use?",
        "top_k": 2,
    }
    response = test_client.post("/questions/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is VeriDoc and what libraries does it use?"
    assert "PyMuPDF" in data["answer"] or "FAISS" in data["answer"]
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["page_number"] in [1, 2]
    assert data["sources"][0]["chunk_id"] >= 1
    assert data["sources"][0]["similarity_score"] is not None


def test_ask_question_before_upload_returns_400(test_client):
    """Test asking a question before any document is uploaded returns 400."""
    payload = {"question": "What is in this document?"}
    response = test_client.post("/questions/ask", json=payload)
    assert response.status_code == 400
    assert "No document has been processed or indexed yet" in response.json()["detail"]


def test_ask_empty_question_returns_422_or_400(test_client, sample_pdf_bytes):
    """Test asking an empty or whitespace question returns validation error."""
    files = {
        "file": ("doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    test_client.post("/documents/upload", files=files)

    # Empty string (violates min_length=1 or custom validation)
    res1 = test_client.post("/questions/ask", json={"question": ""})
    assert res1.status_code in [400, 422]

    # Whitespace only
    res2 = test_client.post("/questions/ask", json={"question": "    "})
    assert res2.status_code == 400


def test_ask_missing_api_key_returns_503(sample_pdf_bytes):
    """Test that missing GEMINI_API_KEY returns 503 Service Unavailable."""
    pipeline = RAGPipeline(api_key=None, llm_client=None)
    set_rag_pipeline(pipeline)
    client = TestClient(app)

    # Ingest document
    files = {
        "file": ("doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    client.post("/documents/upload", files=files)

    with patch.dict("os.environ", {}, clear=True):
        response = client.post("/questions/ask", json={"question": "Valid query?"})
        assert response.status_code == 503
        assert "LLM Configuration Error" in response.json()["detail"]

    set_rag_pipeline(None)


def test_ask_llm_failure_returns_502(sample_pdf_bytes):
    """Test that Gemini API runtime failures return 502 Bad Gateway."""
    failing_client = MagicMock()
    failing_client.models.generate_content.side_effect = RuntimeError("API Rate Limit Exceeded")

    pipeline = RAGPipeline(api_key="key", llm_client=failing_client)
    set_rag_pipeline(pipeline)
    client = TestClient(app)

    files = {
        "file": ("doc.pdf", io.BytesIO(sample_pdf_bytes), "application/pdf")
    }
    client.post("/documents/upload", files=files)

    response = client.post("/questions/ask", json={"question": "Test query?"})
    assert response.status_code == 502
    assert "LLM Generation Error" in response.json()["detail"]

    set_rag_pipeline(None)
