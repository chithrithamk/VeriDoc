"""
VeriDoc — Shared Pytest Configuration and Reusable Fixtures (Phase 13).

Provides centralized, isolated, and deterministic fixtures for unit, integration,
API, database, and frontend tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore

from backend.database.models import DocumentRecord, QuestionRecord
from backend.database.repository import create_document_record, create_question_record
from backend.database.session import Base, get_db
from backend.main import app, get_rag_pipeline, set_rag_pipeline
from backend.models.schemas import SourceCitation, SupportCheckResponse
from backend.services.chunker import DocumentChunk
from backend.services.embeddings import EmbeddedChunk
from backend.services.pdf_processor import ExtractedDocument, PageData
from backend.services.rag_pipeline import RAGPipeline
from backend.services.support_checker import SupportCheckResult
from backend.services.vector_store import FAISSVectorStore, SearchResult


# -----------------------------------------------------------------------------
# 1. Document & PDF Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Creates a 2-page in-memory PDF with sample text."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 72), "VeriDoc is an AI document intelligence platform using PyMuPDF and FAISS.")
    p2 = doc.new_page()
    p2.insert_text(fitz.Point(50, 72), "FastAPI exposes REST endpoints for document ingestion and Q&A.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def multi_page_pdf_bytes() -> bytes:
    """Creates a 3-page in-memory PDF with distinct section content."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text(fitz.Point(50, 72), "Introduction: VeriDoc is an AI document platform. It extracts page-level text accurately.")
    p2 = doc.new_page()
    p2.insert_text(fitz.Point(50, 72), "Methodology: FAISS vector indexing enables sub-millisecond dense retrieval across chunks.")
    p3 = doc.new_page()
    p3.insert_text(fitz.Point(50, 72), "Conclusion: Grounded answer generation uses retrieved context to eliminate hallucinations.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def blank_pdf_bytes() -> bytes:
    """Creates a 1-page in-memory PDF with a blank page."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_extracted_doc() -> ExtractedDocument:
    """Creates a standard 3-page ExtractedDocument object."""
    return ExtractedDocument(
        file_path="FINAL_REPORT.pdf",
        total_pages=3,
        pages=[
            PageData(page_number=1, text="Page 1: Overview of AI document intelligence platform.", char_count=54),
            PageData(page_number=2, text="Page 2: Vector retrieval and chunking strategies in RAG.", char_count=56),
            PageData(page_number=3, text="Page 3: Conclusion and evaluation metrics.", char_count=42),
        ],
        total_characters=152,
        has_text=True,
    )


# -----------------------------------------------------------------------------
# 2. Chunks & Vector Store Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_document_chunks() -> list[DocumentChunk]:
    """Creates a list of 3 DocumentChunk objects with sequential IDs."""
    return [
        DocumentChunk(chunk_id=1, text="Text regarding Document Parsing and Extraction.", page_number=1, document_name="doc.pdf"),
        DocumentChunk(chunk_id=2, text="Text regarding Text Chunking and Tokenization.", page_number=2, document_name="doc.pdf"),
        DocumentChunk(chunk_id=3, text="Text regarding FAISS Vector Indexing.", page_number=3, document_name="doc.pdf"),
    ]


@pytest.fixture
def sample_embedded_chunks() -> list[EmbeddedChunk]:
    """Creates 3 EmbeddedChunk unit vectors in 4D space."""
    return [
        EmbeddedChunk(
            chunk_id=1,
            text="Text regarding Document Parsing and Extraction.",
            page_number=1,
            document_name="architecture_doc.pdf",
            embedding=[1.0, 0.0, 0.0, 0.0],
            char_count=46,
        ),
        EmbeddedChunk(
            chunk_id=2,
            text="Text regarding Text Chunking and Tokenization.",
            page_number=2,
            document_name="architecture_doc.pdf",
            embedding=[0.0, 1.0, 0.0, 0.0],
            char_count=46,
        ),
        EmbeddedChunk(
            chunk_id=3,
            text="Text regarding FAISS Vector Indexing.",
            page_number=3,
            document_name="architecture_doc.pdf",
            embedding=[0.0, 0.0, 1.0, 0.0],
            char_count=37,
        ),
    ]


@pytest.fixture
def built_vector_store(sample_embedded_chunks) -> FAISSVectorStore:
    """Returns a FAISSVectorStore pre-populated with deterministic chunks."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)
    return store


# -----------------------------------------------------------------------------
# 3. LLM Mocks & Support Checker Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_genai_client():
    """Mock Google Gemini client returning predictable grounded response."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "VeriDoc uses PyMuPDF for text extraction and FAISS for vector indexing."
    client.models.generate_content.return_value = mock_response
    return client


@pytest.fixture
def mock_llm_client(mock_genai_client):
    """Alias for mock_genai_client."""
    return mock_genai_client


# -----------------------------------------------------------------------------
# 4. Database Isolation Fixtures (In-Memory SQLite)
# -----------------------------------------------------------------------------

@pytest.fixture
def in_memory_db():
    """Provides a fresh isolated in-memory SQLite database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_db_session(in_memory_db):
    """Alias for in_memory_db."""
    return in_memory_db


# -----------------------------------------------------------------------------
# 5. FastAPI Test Client Fixture
# -----------------------------------------------------------------------------

@pytest.fixture
def api_test_client(in_memory_db, mock_llm_client):
    """Provides a FastAPI TestClient configured with in-memory DB and mocked LLM."""
    def override_get_db():
        try:
            yield in_memory_db
        finally:
            pass

    pipeline = RAGPipeline(api_key="mock-test-key", llm_client=mock_llm_client)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_rag_pipeline] = lambda: pipeline
    set_rag_pipeline(pipeline)

    with TestClient(app) as client:
        yield client

    set_rag_pipeline(None)
    app.dependency_overrides.clear()
    pipeline.clear()
