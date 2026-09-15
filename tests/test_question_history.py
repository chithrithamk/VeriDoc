"""
Tests for Phase 12: Persistent Q&A History (SQLite & SQLAlchemy Integration).

Validates:
1. QuestionRecord ORM model and repository CRUD operations.
2. DocumentRecord and QuestionRecord cascade relationships.
3. POST /questions/ask automatic persistence in SQLite.
4. GET /questions/history with pagination and document filtering.
5. GET /questions/history/{id} and GET /documents/{id}/questions.
6. DELETE /questions/history/{id} and DELETE /questions/history.
7. Support verification details and citation metadata preservation in SQLite.
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
from backend.database.repository import (
    clear_all_document_records,
    clear_all_question_records,
    count_question_records,
    create_document_record,
    create_question_record,
    delete_document_record,
    delete_question_record,
    get_document_record_by_id,
    get_question_record_by_id,
    list_document_records,
    list_question_records,
    list_questions_for_document,
)
from backend.database.session import Base, get_db
from backend.main import app, get_rag_pipeline
from backend.models.schemas import (
    QuestionHistoryItemResponse,
    QuestionHistoryListResponse,
    QuestionResponse,
    SourceCitation,
    SupportCheckResponse,
)
from backend.services.generator import GeneratedAnswer
from backend.services.rag_pipeline import RAGPipeline
from backend.services.support_checker import SupportCheckResult
from backend.services.vector_store import SearchResult
from backend.services.chunker import DocumentChunk


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
def sample_pdf_bytes():
    """Generates sample PDF bytes with 2 pages for testing."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(fitz.Point(50, 72), "VeriDoc provides AI-assisted document intelligence and verification.")
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(50, 72), "Persistent SQLite storage ensures Q&A history and source citations are saved.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def client_with_db(in_memory_db):
    """Provides a TestClient wired to the in-memory SQLite database and a fresh RAG pipeline."""
    def override_get_db():
        try:
            yield in_memory_db
        finally:
            pass

    test_pipeline = RAGPipeline()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_rag_pipeline] = lambda: test_pipeline

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    test_pipeline.clear()


# =============================================================================
# 1. ORM Model & Repository CRUD Tests
# =============================================================================

def test_create_and_get_question_record(in_memory_db):
    """Test creating and retrieving a QuestionRecord with citations and support checks."""
    doc = create_document_record(
        db=in_memory_db,
        filename="test_doc.pdf",
        total_pages=2,
        total_characters=500,
        total_chunks=3,
        indexed_vectors=3,
    )

    sources = [
        {
            "chunk_id": 1,
            "page_number": 1,
            "document_name": "test_doc.pdf",
            "char_count": 120,
            "text": "VeriDoc provides document intelligence.",
            "similarity_score": 0.88,
        }
    ]

    record = create_question_record(
        db=in_memory_db,
        question="What is VeriDoc?",
        answer="VeriDoc is an AI document intelligence platform.",
        document_id=doc.id,
        document_name=doc.filename,
        sources=sources,
        support_status="supported",
        support_confidence=0.95,
        support_explanation="Directly stated on page 1.",
        supported_claims=["VeriDoc is a document intelligence platform."],
        unsupported_claims=[],
    )

    assert record.id is not None
    assert record.question == "What is VeriDoc?"
    assert record.answer == "VeriDoc is an AI document intelligence platform."
    assert record.document_id == doc.id
    assert record.document_name == "test_doc.pdf"
    assert record.support_status == "supported"
    assert record.support_confidence == 0.95
    assert len(record.sources_data) == 1
    assert record.sources_data[0]["page_number"] == 1

    # Fetch by ID
    fetched = get_question_record_by_id(in_memory_db, record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.question == "What is VeriDoc?"

    # to_dict verification
    data = fetched.to_dict()
    assert data["id"] == record.id
    assert data["support"]["status"] == "supported"
    assert data["support"]["confidence"] == 0.95
    assert len(data["sources"]) == 1


from datetime import datetime, timedelta, timezone


def test_list_question_records_pagination(in_memory_db):
    """Test listing question records with ordering and pagination."""
    doc = create_document_record(
        db=in_memory_db,
        filename="doc.pdf",
        total_pages=1,
        total_characters=100,
        total_chunks=1,
        indexed_vectors=1,
    )

    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        create_question_record(
            db=in_memory_db,
            question=f"Question {i}",
            answer=f"Answer {i}",
            document_id=doc.id,
            document_name="doc.pdf",
            created_at=base_time + timedelta(minutes=i),
        )

    all_records = list_question_records(in_memory_db, limit=10)
    assert len(all_records) == 5

    # Reverse chronological order
    assert all_records[0].question == "Question 4"
    assert all_records[-1].question == "Question 0"

    # Pagination: offset and limit
    page1 = list_question_records(in_memory_db, limit=2, offset=0)
    page2 = list_question_records(in_memory_db, limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    assert page1[0].question == "Question 4"
    assert page2[0].question == "Question 2"

    assert count_question_records(in_memory_db) == 5



def test_list_questions_for_specific_document(in_memory_db):
    """Test filtering questions by document ID."""
    doc1 = create_document_record(db=in_memory_db, filename="doc1.pdf", total_pages=1, total_characters=100, total_chunks=1, indexed_vectors=1)
    doc2 = create_document_record(db=in_memory_db, filename="doc2.pdf", total_pages=1, total_characters=100, total_chunks=1, indexed_vectors=1)

    create_question_record(db=in_memory_db, question="Q1 for doc1", answer="A1", document_id=doc1.id, document_name="doc1.pdf")
    create_question_record(db=in_memory_db, question="Q2 for doc1", answer="A2", document_id=doc1.id, document_name="doc1.pdf")
    create_question_record(db=in_memory_db, question="Q1 for doc2", answer="A3", document_id=doc2.id, document_name="doc2.pdf")

    doc1_questions = list_questions_for_document(in_memory_db, document_id=doc1.id)
    doc2_questions = list_questions_for_document(in_memory_db, document_id=doc2.id)

    assert len(doc1_questions) == 2
    assert len(doc2_questions) == 1
    assert count_question_records(in_memory_db, document_id=doc1.id) == 2
    assert count_question_records(in_memory_db, document_id=doc2.id) == 1


def test_delete_question_record(in_memory_db):
    """Test deleting a single question record."""
    q = create_question_record(db=in_memory_db, question="Delete me?", answer="Yes")
    assert count_question_records(in_memory_db) == 1

    success = delete_question_record(in_memory_db, q.id)
    assert success is True
    assert count_question_records(in_memory_db) == 0

    # Non-existent ID returns False
    assert delete_question_record(in_memory_db, "non-existent-uuid") is False


def test_clear_all_question_records_preserves_documents(in_memory_db):
    """Test that clearing question records does NOT delete documents."""
    doc = create_document_record(db=in_memory_db, filename="doc.pdf", total_pages=1, total_characters=100, total_chunks=1, indexed_vectors=1)
    create_question_record(db=in_memory_db, question="Q1", answer="A1", document_id=doc.id)
    create_question_record(db=in_memory_db, question="Q2", answer="A2", document_id=doc.id)

    assert count_question_records(in_memory_db) == 2
    assert len(list_document_records(in_memory_db)) == 1

    deleted_count = clear_all_question_records(in_memory_db)
    assert deleted_count == 2
    assert count_question_records(in_memory_db) == 0

    # Document must still exist!
    assert get_document_record_by_id(in_memory_db, doc.id) is not None


def test_document_deletion_cascades_to_questions(in_memory_db):
    """Test that deleting a document cascades and deletes all its questions."""
    doc = create_document_record(db=in_memory_db, filename="doc.pdf", total_pages=1, total_characters=100, total_chunks=1, indexed_vectors=1)
    create_question_record(db=in_memory_db, question="Q1", answer="A1", document_id=doc.id)
    create_question_record(db=in_memory_db, question="Q2", answer="A2", document_id=doc.id)

    assert count_question_records(in_memory_db) == 2

    delete_document_record(in_memory_db, doc.id)
    assert count_question_records(in_memory_db) == 0


# =============================================================================
# 2. FastAPI Endpoints Integration Tests
# =============================================================================

def test_ask_question_persists_in_database(client_with_db, sample_pdf_bytes, in_memory_db):
    """Test that POST /questions/ask stores the interaction and support results in SQLite."""
    # 1. Upload PDF
    upload_resp = client_with_db.post(
        "/documents/upload",
        files={"file": ("test_paper.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert upload_resp.status_code == 201
    doc_id = upload_resp.json()["id"]

    # 2. Mock Gemini answer & support checker
    with patch("backend.services.generator.AnswerGenerator.generate_answer") as mock_gen, \
         patch("backend.services.support_checker.SupportChecker.check_support") as mock_supp:

        mock_gen.return_value = GeneratedAnswer(
            question="What does VeriDoc provide?",
            answer="VeriDoc provides AI-assisted document intelligence and verification.",
            sources=[
                SearchResult(
                    chunk=DocumentChunk(
                        chunk_id=1,
                        page_number=1,
                        text="VeriDoc provides AI-assisted document intelligence and verification.",
                        char_count=69,
                        document_name="test_paper.pdf",
                    ),
                    score=0.92,
                )
            ],
        )

        mock_supp.return_value = SupportCheckResult(
            status="supported",
            confidence=0.96,
            explanation="Verbatim statement from page 1.",
            supported_claims=["VeriDoc provides AI-assisted document intelligence."],
            unsupported_claims=[],
        )

        # 3. Ask question
        ask_resp = client_with_db.post(
            "/questions/ask",
            json={"question": "What does VeriDoc provide?", "top_k": 3},
        )
        assert ask_resp.status_code == 200
        ask_data = ask_resp.json()

        assert ask_data["id"] is not None
        assert ask_data["document_id"] == doc_id
        assert ask_data["document_name"] == "test_paper.pdf"
        assert ask_data["question"] == "What does VeriDoc provide?"
        assert ask_data["support"]["status"] == "supported"
        assert ask_data["support"]["confidence"] == 0.96

        # 4. Verify SQLite has exactly 1 question record with correct fields
        records = list_question_records(in_memory_db)
        assert len(records) == 1
        assert records[0].id == ask_data["id"]
        assert records[0].document_id == doc_id
        assert records[0].document_name == "test_paper.pdf"
        assert records[0].support_status == "supported"
        assert records[0].support_confidence == 0.96
        assert len(records[0].sources_data) == 1
        assert records[0].sources_data[0]["page_number"] == 1


def test_get_question_history_endpoints(client_with_db, sample_pdf_bytes):
    """Test GET /questions/history and GET /questions/history/{id}."""
    # Upload
    client_with_db.post(
        "/documents/upload",
        files={"file": ("doc.pdf", sample_pdf_bytes, "application/pdf")},
    )

    with patch("backend.services.generator.AnswerGenerator.generate_answer") as mock_gen, \
         patch("backend.services.support_checker.SupportChecker.check_support") as mock_supp:

        mock_gen.side_effect = lambda question, sources, **kwargs: GeneratedAnswer(
            question=question,
            answer=f"Answer to {question}",
            sources=[],
        )
        mock_supp.return_value = SupportCheckResult(
            status="insufficient_evidence",
            confidence=0.0,
            explanation="No sources.",
            supported_claims=[],
            unsupported_claims=[],
        )

        # Ask 2 questions
        r1 = client_with_db.post("/questions/ask", json={"question": "First question?"})
        assert r1.status_code == 200
        q1_id = r1.json()["id"]

        r2 = client_with_db.post("/questions/ask", json={"question": "Second question?"})
        assert r2.status_code == 200
        q2_id = r2.json()["id"]


    # Test GET /questions/history
    hist_resp = client_with_db.get("/questions/history?limit=10&offset=0")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["total"] == 2
    assert len(hist_data["questions"]) == 2
    # Reverse chronological
    assert hist_data["questions"][0]["id"] == q2_id
    assert hist_data["questions"][1]["id"] == q1_id

    # Test GET /questions/history/{id}
    q1_resp = client_with_db.get(f"/questions/history/{q1_id}")
    assert q1_resp.status_code == 200
    assert q1_resp.json()["id"] == q1_id
    assert q1_resp.json()["question"] == "First question?"

    # Non-existent ID returns 404
    not_found = client_with_db.get("/questions/history/non-existent-id")
    assert not_found.status_code == 404


def test_get_document_questions_endpoint(client_with_db, sample_pdf_bytes):
    """Test GET /documents/{doc_id}/questions endpoint."""
    upload_resp = client_with_db.post(
        "/documents/upload",
        files={"file": ("doc.pdf", sample_pdf_bytes, "application/pdf")},
    )
    doc_id = upload_resp.json()["id"]

    with patch("backend.services.generator.AnswerGenerator.generate_answer") as mock_gen, \
         patch("backend.services.support_checker.SupportChecker.check_support") as mock_supp:

        mock_gen.return_value = GeneratedAnswer(question="Q?", answer="A.", sources=[])
        mock_supp.return_value = SupportCheckResult(status="supported", confidence=0.9, explanation="", supported_claims=[], unsupported_claims=[])

        client_with_db.post("/questions/ask", json={"question": "Q?"})

    doc_q_resp = client_with_db.get(f"/documents/{doc_id}/questions")
    assert doc_q_resp.status_code == 200
    assert doc_q_resp.json()["total"] == 1
    assert len(doc_q_resp.json()["questions"]) == 1

    # Non-existent doc_id returns 404
    inv_doc = client_with_db.get("/documents/invalid-id/questions")
    assert inv_doc.status_code == 404


def test_delete_question_endpoints(client_with_db, sample_pdf_bytes):
    """Test DELETE /questions/history/{id} and DELETE /questions/history."""
    client_with_db.post(
        "/documents/upload",
        files={"file": ("doc.pdf", sample_pdf_bytes, "application/pdf")},
    )

    with patch("backend.services.generator.AnswerGenerator.generate_answer") as mock_gen, \
         patch("backend.services.support_checker.SupportChecker.check_support") as mock_supp:

        mock_gen.return_value = GeneratedAnswer(question="Q1", answer="A1", sources=[])
        mock_supp.return_value = SupportCheckResult(status="supported", confidence=1.0, explanation="", supported_claims=[], unsupported_claims=[])

        r1 = client_with_db.post("/questions/ask", json={"question": "Q1"})
        r2 = client_with_db.post("/questions/ask", json={"question": "Q2"})
        q1_id = r1.json()["id"]

    # Delete single record
    del1 = client_with_db.delete(f"/questions/history/{q1_id}")
    assert del1.status_code == 200
    assert del1.json()["status"] == "success"

    # Verify 1 remains
    hist = client_with_db.get("/questions/history")
    assert hist.json()["total"] == 1

    # Delete all records
    del_all = client_with_db.delete("/questions/history")
    assert del_all.status_code == 200
    assert del_all.json()["deleted_count"] == 1

    # Verify 0 remains
    hist2 = client_with_db.get("/questions/history")
    assert hist2.json()["total"] == 0
