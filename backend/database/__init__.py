"""
VeriDoc — Database Package (Phase 9)
"""

from backend.database.models import DocumentRecord
from backend.database.session import Base, engine, get_db, init_db
from backend.database.repository import (
    clear_all_document_records,
    create_document_record,
    delete_document_record,
    get_document_record_by_id,
    get_latest_document_record,
    list_document_records,
)

__all__ = [
    "Base",
    "engine",
    "get_db",
    "init_db",
    "DocumentRecord",
    "create_document_record",
    "get_document_record_by_id",
    "get_latest_document_record",
    "list_document_records",
    "delete_document_record",
    "clear_all_document_records",
]
