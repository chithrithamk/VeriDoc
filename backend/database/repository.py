"""
VeriDoc — Document and Question Database Repositories (Phases 9 & 12)

Encapsulates data access and persistence operations for DocumentRecord and QuestionRecord entities.
"""

from typing import Any, Dict, List, Optional
from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.database.models import DocumentRecord, QuestionRecord


# -----------------------------------------------------------------------------
# Document Repository Operations
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Question & Q&A History Repository Operations
# -----------------------------------------------------------------------------

def create_question_record(
    db: Session,
    question: str,
    answer: str,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    support_status: Optional[str] = None,
    support_confidence: Optional[float] = None,
    support_explanation: Optional[str] = None,
    supported_claims: Optional[List[str]] = None,
    unsupported_claims: Optional[List[str]] = None,
    created_at: Optional[Any] = None,
) -> QuestionRecord:
    """
    Creates and persists a new QuestionRecord in SQLite.
    """
    record = QuestionRecord(
        document_id=document_id,
        document_name=document_name,
        question=question,
        answer=answer,
        sources_data=sources or [],
        support_status=support_status,
        support_confidence=support_confidence,
        support_explanation=support_explanation,
        supported_claims=supported_claims or [],
        unsupported_claims=unsupported_claims or [],
    )
    if created_at is not None:
        record.created_at = created_at
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_question_record_by_id(db: Session, question_id: str) -> Optional[QuestionRecord]:
    """Retrieves a single question record by its primary key ID."""
    return db.query(QuestionRecord).filter(QuestionRecord.id == question_id).first()


def list_question_records(
    db: Session,
    limit: int = 50,
    offset: int = 0,
) -> List[QuestionRecord]:
    """Lists all question records in reverse chronological order."""
    return (
        db.query(QuestionRecord)
        .order_by(desc(QuestionRecord.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )


def list_questions_for_document(
    db: Session,
    document_id: str,
    limit: int = 50,
    offset: int = 0,
) -> List[QuestionRecord]:
    """Lists question records associated with a specific document ID."""
    return (
        db.query(QuestionRecord)
        .filter(QuestionRecord.document_id == document_id)
        .order_by(desc(QuestionRecord.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )


def count_question_records(db: Session, document_id: Optional[str] = None) -> int:
    """Returns the total number of question records (optionally filtered by document_id)."""
    query = db.query(QuestionRecord)
    if document_id:
        query = query.filter(QuestionRecord.document_id == document_id)
    return query.count()


def delete_question_record(db: Session, question_id: str) -> bool:
    """Deletes a question record by ID. Returns True if deleted, False if not found."""
    record = get_question_record_by_id(db, question_id)
    if record:
        db.delete(record)
        db.commit()
        return True
    return False


def clear_all_question_records(db: Session, document_id: Optional[str] = None) -> int:
    """
    Deletes question records. If document_id is specified, deletes only questions for that document.
    Otherwise, deletes all question records across all documents.
    """
    query = db.query(QuestionRecord)
    if document_id:
        query = query.filter(QuestionRecord.document_id == document_id)
    count = query.delete(synchronize_session=False)
    db.commit()
    return count
