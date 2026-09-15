"""
VeriDoc — SQLAlchemy ORM Models (Phases 9 & 12)

Defines relational database schemas for:
1. DocumentRecord: Persistent document metadata (PDF pages, chunks, vectors, status).
2. QuestionRecord: Persistent Q&A history (questions, answers, source citations, support checks).
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from backend.database.session import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(Base):
    """
    SQLAlchemy ORM model representing persistent metadata for an ingested PDF document.
    """
    __tablename__ = "documents"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    filename = Column(String(255), nullable=False, index=True)
    file_path = Column(String(512), nullable=True)
    file_size_bytes = Column(Integer, nullable=True, default=0)
    total_pages = Column(Integer, nullable=False, default=0)
    total_characters = Column(Integer, nullable=False, default=0)
    total_chunks = Column(Integer, nullable=False, default=0)
    indexed_vectors = Column(Integer, nullable=False, default=0)
    status = Column(String(50), nullable=False, default="indexed", index=True)
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)

    # Relationship to questions (cascading on delete)
    questions = relationship("QuestionRecord", back_populates="document", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Serializes model instance into a dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "total_pages": self.total_pages,
            "total_characters": self.total_characters,
            "total_chunks": self.total_chunks,
            "indexed_vectors": self.indexed_vectors,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class QuestionRecord(Base):
    """
    SQLAlchemy ORM model representing persistent Q&A interaction with citations and support checks.
    """
    __tablename__ = "questions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    document_id = Column(String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=True, index=True)
    document_name = Column(String(255), nullable=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    support_status = Column(String(50), nullable=True, default="verification_unavailable")
    support_confidence = Column(Float, nullable=True, default=0.0)
    support_explanation = Column(Text, nullable=True)
    supported_claims = Column(JSON, nullable=True, default=list)
    unsupported_claims = Column(JSON, nullable=True, default=list)
    sources_data = Column(JSON, nullable=True, default=list)
    created_at = Column(DateTime, default=_utc_now, nullable=False)
    updated_at = Column(DateTime, default=_utc_now, onupdate=_utc_now, nullable=False)

    # Relationship to document
    document = relationship("DocumentRecord", back_populates="questions")

    def to_dict(self) -> dict:
        """Serializes question record into a dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "question": self.question,
            "answer": self.answer,
            "support": {
                "status": self.support_status or "verification_unavailable",
                "confidence": float(self.support_confidence) if self.support_confidence is not None else 0.0,
                "explanation": self.support_explanation or "",
                "supported_claims": list(self.supported_claims) if self.supported_claims else [],
                "unsupported_claims": list(self.unsupported_claims) if self.unsupported_claims else [],
            } if self.support_status else None,
            "sources": list(self.sources_data) if self.sources_data else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
