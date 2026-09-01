"""
Unit and integration tests for SQLite Database Layer and Repositories (Phase 9).
"""

from datetime import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database.models import DocumentRecord
from backend.database.repository import (
    clear_all_document_records,
    create_document_record,
    delete_document_record,
    get_document_record_by_id,
    get_latest_document_record,
    list_document_records,
)
from backend.database.session import Base, init_db


@pytest.fixture
def test_db_session():
    """Creates an isolated in-memory SQLite database session for unit testing."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


def test_init_db_creates_tables():
    """Test that init_db() creates tables on target engine without error."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(target_engine=test_engine)
    # Re-running init_db should be idempotent
    init_db(target_engine=test_engine)


def test_create_and_get_document_record(test_db_session):
    """Test creating a DocumentRecord and retrieving it by ID."""
    record = create_document_record(
        db=test_db_session,
        filename="ai_paper.pdf",
        total_pages=5,
        total_characters=4500,
        total_chunks=8,
        indexed_vectors=8,
        file_size_bytes=10240,
        status="indexed",
    )

    assert record.id is not None
    assert len(record.id) == 36  # UUID string
    assert record.filename == "ai_paper.pdf"
    assert record.total_pages == 5
    assert record.total_characters == 4500
    assert record.total_chunks == 8
    assert record.indexed_vectors == 8
    assert record.status == "indexed"
    assert isinstance(record.created_at, datetime)

    # Retrieve by ID
    fetched = get_document_record_by_id(test_db_session, record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.filename == "ai_paper.pdf"


def test_get_latest_document_record(test_db_session):
    """Test retrieving the latest uploaded document."""
    # When empty
    assert get_latest_document_record(test_db_session) is None

    # Insert two documents
    doc1 = create_document_record(
        db=test_db_session,
        filename="first.pdf",
        total_pages=1,
        total_characters=100,
        total_chunks=1,
        indexed_vectors=1,
    )
    doc2 = create_document_record(
        db=test_db_session,
        filename="second.pdf",
        total_pages=2,
        total_characters=200,
        total_chunks=2,
        indexed_vectors=2,
    )

    latest = get_latest_document_record(test_db_session)
    assert latest is not None
    assert latest.id == doc2.id
    assert latest.filename == "second.pdf"


def test_list_document_records_pagination(test_db_session):
    """Test listing documents with limit and offset pagination."""
    for i in range(5):
        create_document_record(
            db=test_db_session,
            filename=f"doc_{i}.pdf",
            total_pages=i + 1,
            total_characters=(i + 1) * 100,
            total_chunks=i + 1,
            indexed_vectors=i + 1,
        )

    all_docs = list_document_records(test_db_session, limit=10)
    assert len(all_docs) == 5

    page1 = list_document_records(test_db_session, limit=2, offset=0)
    assert len(page1) == 2

    page2 = list_document_records(test_db_session, limit=2, offset=2)
    assert len(page2) == 2
    assert page1[0].id != page2[0].id


def test_delete_document_record(test_db_session):
    """Test deleting a document record by ID."""
    doc = create_document_record(
        db=test_db_session,
        filename="temp.pdf",
        total_pages=1,
        total_characters=100,
        total_chunks=1,
        indexed_vectors=1,
    )

    # Successful delete
    deleted = delete_document_record(test_db_session, doc.id)
    assert deleted is True
    assert get_document_record_by_id(test_db_session, doc.id) is None

    # Delete non-existent ID
    deleted_again = delete_document_record(test_db_session, "non-existent-uuid")
    assert deleted_again is False


def test_clear_all_document_records(test_db_session):
    """Test clearing all rows in documents table."""
    create_document_record(test_db_session, "a.pdf", 1, 100, 1, 1)
    create_document_record(test_db_session, "b.pdf", 2, 200, 2, 2)

    count = clear_all_document_records(test_db_session)
    assert count == 2
    assert len(list_document_records(test_db_session)) == 0


def test_document_record_to_dict(test_db_session):
    """Test DocumentRecord serialization method."""
    record = create_document_record(
        db=test_db_session,
        filename="sample.pdf",
        total_pages=3,
        total_characters=1200,
        total_chunks=4,
        indexed_vectors=4,
        file_size_bytes=5000,
    )
    d = record.to_dict()
    assert d["id"] == record.id
    assert d["filename"] == "sample.pdf"
    assert d["total_pages"] == 3
    assert d["status"] == "indexed"
    assert d["created_at"] is not None
