"""
VeriDoc — Database Package (Phases 9 & 12)
"""

from backend.database.models import DocumentRecord, QuestionRecord
from backend.database.session import Base, engine, get_db, init_db
from backend.database.repository import (
    clear_all_document_records,
    clear_all_question_records,
    count_question_records,
    create_document_record,
    create_question_record,
    delete_document_record,
    delete_question_record,
    get_document_record_by_id,
    get_latest_document_record,
    get_question_record_by_id,
    list_document_records,
    list_question_records,
    list_questions_for_document,
)

__all__ = [
    "Base",
    "engine",
    "get_db",
    "init_db",
    "DocumentRecord",
    "QuestionRecord",
    "create_document_record",
    "get_document_record_by_id",
    "get_latest_document_record",
    "list_document_records",
    "delete_document_record",
    "clear_all_document_records",
    "create_question_record",
    "get_question_record_by_id",
    "list_question_records",
    "list_questions_for_document",
    "count_question_records",
    "delete_question_record",
    "clear_all_question_records",
]
