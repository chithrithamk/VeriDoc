"""
VeriDoc — PDF Processing Service (Phase 1)

This module handles opening PDF documents, validating input files, and extracting
structured text on a per-page basis using PyMuPDF (fitz).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class PDFProcessingError(Exception):
    """Base exception for all PDF processing errors."""
    pass


class PDFNotFoundError(PDFProcessingError, FileNotFoundError):
    """Raised when the specified PDF file cannot be found."""
    pass


class InvalidPDFError(PDFProcessingError, ValueError):
    """Raised when the input file is not a valid PDF or has an invalid format."""
    pass


class CorruptedPDFError(PDFProcessingError):
    """Raised when a PDF file is corrupted or cannot be parsed."""
    pass


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class PageData:
    """Represents extracted text and metadata for an individual PDF page."""
    page_number: int  # 1-indexed (human-readable)
    text: str
    char_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert page data to a dictionary."""
        return {
            "page_number": self.page_number,
            "text": self.text,
            "char_count": self.char_count,
        }


@dataclass
class ExtractedDocument:
    """Represents the complete extracted content and page-level breakdown of a PDF."""
    file_path: str
    total_pages: int
    pages: List[PageData] = field(default_factory=list)
    total_characters: int = 0
    has_text: bool = False

    def get_full_text(self, separator: str = "\n\n") -> str:
        """Combine extracted text across all pages using a configurable separator."""
        return separator.join(page.text for page in self.pages if page.text)

    def get_page(self, page_number: int) -> Optional[PageData]:
        """Retrieve a specific page by its 1-indexed page number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entire extracted document to a dictionary."""
        return {
            "file_path": self.file_path,
            "total_pages": self.total_pages,
            "total_characters": self.total_characters,
            "has_text": self.has_text,
            "pages": [page.to_dict() for page in self.pages],
        }


# -----------------------------------------------------------------------------
# Extraction Functions
# -----------------------------------------------------------------------------

def extract_text_from_pdf(file_path: Union[str, Path]) -> ExtractedDocument:
    """
    Extracts text page-by-page from a PDF document.

    Args:
        file_path: Path to the PDF file (string or pathlib.Path).

    Returns:
        ExtractedDocument: A structured object containing per-page text and metadata.

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidPDFError: If the file extension is not .pdf.
        CorruptedPDFError: If PyMuPDF fails to open or parse the PDF document.
        PDFProcessingError: For unexpected errors during processing.
    """
    path = Path(file_path)

    # 1. Check file existence
    if not path.exists():
        raise PDFNotFoundError(f"PDF file does not exist: {path}")
    if not path.is_file():
        raise PDFNotFoundError(f"Provided path is not a regular file: {path}")

    # 2. Validate file extension
    if path.suffix.lower() != ".pdf":
        raise InvalidPDFError(
            f"Invalid file format '{path.suffix}'. Only .pdf files are supported."
        )

    # 3. Open PDF document with PyMuPDF
    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise CorruptedPDFError(f"Unable to open PDF file '{path}': {exc}") from exc

    try:
        # Check for password/encryption
        if doc.is_encrypted:
            raise PDFProcessingError(
                f"Password-protected or encrypted PDFs are not supported: '{path}'"
            )

        total_pages = len(doc)
        pages: List[PageData] = []
        total_chars = 0

        # 4. Extract text from each page
        for page_index in range(total_pages):
            page = doc[page_index]
            page_text = page.get_text() or ""
            cleaned_text = page_text.strip()
            char_count = len(cleaned_text)
            total_chars += char_count

            pages.append(
                PageData(
                    page_number=page_index + 1,  # 1-indexed for citations
                    text=cleaned_text,
                    char_count=char_count,
                )
            )

        has_text = any(len(p.text) > 0 for p in pages)

        return ExtractedDocument(
            file_path=str(path),
            total_pages=total_pages,
            pages=pages,
            total_characters=total_chars,
            has_text=has_text,
        )

    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            f"An error occurred while extracting text from '{path}': {exc}"
        ) from exc
    finally:
        doc.close()
