"""
VeriDoc — SQLAlchemy ORM Models (Phase 9)

Defines relational database schemas for persistent document metadata.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, DateTime, Integer, String

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
