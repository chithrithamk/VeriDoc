"""
VeriDoc — Document Database Repository (Phase 9)

Encapsulates data access and persistence operations for DocumentRecord entities.
"""

from typing import List, Optional
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.database.models import DocumentRecord


def create_document_record(
    db: Session,
    filename: str,
    total_pages: int,
    total_characters: int,
    total_chunks: int,
    indexed_vectors: int,
    file_path: Optional[str] = None,
    file_size_bytes: int = 0,
    status: str = "indexed",
) -> DocumentRecord:
    """
    Creates and persists a new DocumentRecord in SQLite.

    Args:
        db: Active SQLAlchemy database session.
        filename: Name of the uploaded file.
        total_pages: Extracted page count.
        total_characters: Extracted character count.
        total_chunks: Number of chunks created.
        indexed_vectors: Number of vectors indexed in FAISS.
        file_path: Local storage path of the document.
        file_size_bytes: Byte size of the uploaded file.
        status: Ingestion status string (e.g. "indexed", "failed").

    Returns:
        DocumentRecord: Persisted ORM document record.
    """
    record = DocumentRecord(
        filename=filename,
        file_path=file_path,
        file_size_bytes=file_size_bytes,
        total_pages=total_pages,
        total_characters=total_characters,
        total_chunks=total_chunks,
        indexed_vectors=indexed_vectors,
        status=status,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_document_record_by_id(db: Session, doc_id: str) -> Optional[DocumentRecord]:
    """Retrieves a document record by its primary key ID."""
    return db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()


def get_latest_document_record(db: Session) -> Optional[DocumentRecord]:
    """Retrieves the most recently created document record."""
    return db.query(DocumentRecord).order_by(desc(DocumentRecord.created_at)).first()


def list_document_records(
    db: Session,
    limit: int = 50,
    offset: int = 0,
) -> List[DocumentRecord]:
    """Lists document records in reverse chronological order."""
    return (
        db.query(DocumentRecord)
        .order_by(desc(DocumentRecord.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )


def delete_document_record(db: Session, doc_id: str) -> bool:
    """Deletes a document record by ID. Returns True if deleted, False if not found."""
    record = get_document_record_by_id(db, doc_id)
    if record:
        db.delete(record)
        db.commit()
        return True
    return False


def clear_all_document_records(db: Session) -> int:
    """Deletes all document records from the database. Returns count of deleted rows."""
    count = db.query(DocumentRecord).delete()
    db.commit()
    return count
