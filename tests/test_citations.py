"""
Comprehensive tests for Phase 10: Source & Page Citations and Multi-Question Q&A UX.
"""

import io
from pathlib import Path
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
from streamlit.testing.v1 import AppTest

from backend.database.session import Base, get_db
from backend.main import app, set_rag_pipeline
from backend.models.schemas import QuestionResponse, SourceCitation
from backend.services.chunker import DocumentChunk
from backend.services.embeddings import EmbeddedChunk, embed_chunks
from backend.services.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    LLMConfigurationError,
    LLMGenerationError,
    generate_rag_answer,
)
from backend.services.pdf_processor import ExtractedDocument, PageData
from backend.services.rag_pipeline import RAGPipeline
from backend.services.retrieval import RetrievalService
from backend.services.vector_store import FAISSVectorStore, SearchResult

FRONTEND_APP_PATH = Path(__file__).parent.parent / "frontend" / "app.py"


@pytest.fixture
def multi_page_extracted_doc():
    """Create a 3-page extracted document with distinct content per page."""
    return ExtractedDocument(
        file_path="research_report.pdf",
        total_pages=3,
        pages=[
            PageData(
                page_number=1,
                text="Introduction: VeriDoc is an AI document platform. It extracts page-level text accurately.",
                char_count=87,
            ),
            PageData(
                page_number=2,
                text="Methodology: FAISS vector indexing enables sub-millisecond dense retrieval across chunks.",
                char_count=90,
            ),
            PageData(
                page_number=3,
                text="Conclusion: Grounded answer generation uses retrieved context to eliminate hallucinations.",
                char_count=92,
            ),
        ],
        total_characters=269,
        has_text=True,
    )


@pytest.fixture
def multi_page_pdf_bytes():
    """Create a 3-page in-memory PDF matching multi_page_extracted_doc."""
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
def mock_gemini_client():
    """Mock Gemini client returning predictable grounded response."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "VeriDoc uses FAISS vector indexing for fast semantic retrieval."
    client.models.generate_content.return_value = mock_response
    return client


# -----------------------------------------------------------------------------
# 1. Source & Page Citations Metadata Preservation Tests
# -----------------------------------------------------------------------------

def test_citations_preserve_exact_pdf_page_metadata(multi_page_extracted_doc, mock_gemini_client):
    """Test that citations contain original page numbers, document names, chunk IDs, and similarity scores."""
    pipeline = RAGPipeline(api_key="mock-key", llm_client=mock_gemini_client)
    pipeline.ingest_document(multi_page_extracted_doc)

    answer_result = pipeline.ask("What is the methodology of VeriDoc?", top_k=2)

    assert answer_result.question == "What is the methodology of VeriDoc?"
    assert len(answer_result.sources) == 2

    # Check top citation
    top_citation = answer_result.sources[0]
    assert isinstance(top_citation, SearchResult)
    assert top_citation.chunk.document_name == "research_report.pdf"
    assert top_citation.chunk.page_number in [1, 2, 3]
    assert top_citation.chunk.chunk_id in [1, 2, 3]
    assert isinstance(top_citation.score, float)
    assert top_citation.score >= 0.0
    assert len(top_citation.chunk.text) > 0


def test_citations_not_invented_by_gemini(multi_page_extracted_doc, mock_gemini_client):
    """Verify that citations originate strictly from FAISS retrieval and not the LLM."""
    pipeline = RAGPipeline(api_key="mock-key", llm_client=mock_gemini_client)
    pipeline.ingest_document(multi_page_extracted_doc)

    retrieved_chunks = pipeline.retrieve("FAISS vector indexing", top_k=2)
    answer_result = pipeline.ask("FAISS vector indexing", top_k=2)

    retrieved_ids = [r.chunk.chunk_id for r in retrieved_chunks]
    citation_ids = [s.chunk.chunk_id for s in answer_result.sources]

    assert citation_ids == retrieved_ids
    for src in answer_result.sources:
        assert src.chunk.text in [
            multi_page_extracted_doc.pages[0].text,
            multi_page_extracted_doc.pages[1].text,
            multi_page_extracted_doc.pages[2].text,
        ]


# -----------------------------------------------------------------------------
# 2. Structured API Citations Tests
# -----------------------------------------------------------------------------

def test_api_ask_returns_structured_citations(multi_page_pdf_bytes, mock_gemini_client):
    """Test POST /questions/ask returns valid SourceCitation objects with all metadata fields."""
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
    pipeline = RAGPipeline(api_key="mock-key", llm_client=mock_gemini_client)
    set_rag_pipeline(pipeline)

    client = TestClient(app)

    # Ingest document
    files = {"file": ("report.pdf", io.BytesIO(multi_page_pdf_bytes), "application/pdf")}
    upload_res = client.post("/documents/upload", files=files)
    assert upload_res.status_code == 201

    # Ask question
    qa_res = client.post("/questions/ask", json={"question": "What is the conclusion?", "top_k": 3})
    assert qa_res.status_code == 200
    data = qa_res.json()

    assert data["question"] == "What is the conclusion?"
    assert "answer" in data
    assert len(data["sources"]) == 3

    for src in data["sources"]:
        assert "document_name" in src
        assert src["document_name"] == "report.pdf"
        assert "page_number" in src
        assert src["page_number"] in [1, 2, 3]
        assert "chunk_id" in src
        assert "text" in src
        assert "similarity_score" in src
        assert isinstance(src["similarity_score"], float)

    set_rag_pipeline(None)
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


# -----------------------------------------------------------------------------
# 3. Multi-Question Support & State Preservation Tests
# -----------------------------------------------------------------------------

def test_multiple_questions_against_same_indexed_document(multi_page_extracted_doc, mock_gemini_client):
    """Test asking multiple consecutive questions against the same pipeline without re-ingesting."""
    pipeline = RAGPipeline(api_key="mock-key", llm_client=mock_gemini_client)
    pipeline.ingest_document(multi_page_extracted_doc)

    # First question
    ans1 = pipeline.ask("What does VeriDoc do?", top_k=2)
    assert ans1.question == "What does VeriDoc do?"
    assert len(ans1.sources) == 2

    # Second question (pipeline still ready, index untouched)
    assert pipeline.is_ready() is True
    ans2 = pipeline.ask("What is used for vector search?", top_k=1)
    assert ans2.question == "What is used for vector search?"
    assert len(ans2.sources) == 1

    # Third question
    ans3 = pipeline.ask("What does the conclusion state?", top_k=2)
    assert ans3.question == "What does the conclusion state?"
    assert len(ans3.sources) == 2


def test_processing_new_document_resets_qa_history(multi_page_extracted_doc):
    """Test that resetting or clearing the pipeline resets active document and state."""
    pipeline = RAGPipeline(api_key="mock-key")
    pipeline.ingest_document(multi_page_extracted_doc)
    assert pipeline.is_ready() is True

    # Clear pipeline
    pipeline.clear()
    assert pipeline.is_ready() is False
    assert pipeline.get_stats()["is_ready"] is False
    assert pipeline.get_stats()["total_chunks"] == 0


# -----------------------------------------------------------------------------
# 4. Error Resilience Tests
# -----------------------------------------------------------------------------

def test_gemini_failure_preserves_document_index(multi_page_extracted_doc):
    """Test that an LLM API error raises LLMGenerationError without corrupting the FAISS index."""
    failing_client = MagicMock()
    failing_client.models.generate_content.side_effect = RuntimeError("Quota Exceeded (429)")

    pipeline = RAGPipeline(api_key="mock-key", llm_client=failing_client)
    pipeline.ingest_document(multi_page_extracted_doc)

    # Attempt question that fails at LLM stage
    with pytest.raises(LLMGenerationError):
        pipeline.ask("Explain VeriDoc architecture")

    # Verify pipeline remains ready and vector store intact
    assert pipeline.is_ready() is True
    assert len(pipeline.vector_store) == 3
    retrieved = pipeline.retrieve("VeriDoc", top_k=2)
    assert len(retrieved) == 2
