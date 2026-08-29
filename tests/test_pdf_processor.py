"""
Tests for PDF processing service (Phase 1).
"""

from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore
import pytest

from backend.services.pdf_processor import (
    CorruptedPDFError,
    ExtractedDocument,
    InvalidPDFError,
    PageData,
    PDFNotFoundError,
    PDFProcessingError,
    extract_text_from_pdf,
)


@pytest.fixture
def create_sample_pdf(tmp_path: Path):
    """Factory fixture to create temporary sample PDFs with custom content."""
    def _create(pages_text: list[str], filename: str = "sample.pdf") -> Path:
        file_path = tmp_path / filename
        doc = fitz.open()
        for text in pages_text:
            page = doc.new_page()
            # Insert text at top-left margin (point (50, 72))
            page.insert_text(fitz.Point(50, 72), text)
        doc.save(str(file_path))
        doc.close()
        return file_path

    return _create


# -----------------------------------------------------------------------------
# Success Cases
# -----------------------------------------------------------------------------

def test_extract_text_single_page(create_sample_pdf):
    """Test extracting text from a single-page PDF document."""
    pdf_path = create_sample_pdf(["Hello, this is VeriDoc Phase 1!"])

    result = extract_text_from_pdf(pdf_path)

    assert isinstance(result, ExtractedDocument)
    assert result.total_pages == 1
    assert result.has_text is True
    assert len(result.pages) == 1

    page = result.pages[0]
    assert isinstance(page, PageData)
    assert page.page_number == 1
    assert "Hello, this is VeriDoc Phase 1!" in page.text
    assert page.char_count > 0


def test_extract_text_multi_page(create_sample_pdf):
    """Test extracting text and preserving page-level information across multiple pages."""
    pages_content = [
        "Page 1: Introduction to Retrieval-Augmented Generation.",
        "Page 2: Vector embeddings and FAISS index overview.",
        "Page 3: Conclusion and future scope.",
    ]
    pdf_path = create_sample_pdf(pages_content, filename="multipage.pdf")

    result = extract_text_from_pdf(pdf_path)

    assert result.total_pages == 3
    assert len(result.pages) == 3
    assert result.has_text is True

    for i, expected_text in enumerate(pages_content, start=1):
        page = result.get_page(i)
        assert page is not None
        assert page.page_number == i
        assert expected_text in page.text

    full_text = result.get_full_text()
    assert "Page 1: Introduction" in full_text
    assert "Page 2: Vector embeddings" in full_text
    assert "Page 3: Conclusion" in full_text


def test_extracted_document_to_dict(create_sample_pdf):
    """Test serialization of ExtractedDocument to dictionary format."""
    pdf_path = create_sample_pdf(["Document metadata test"])

    result = extract_text_from_pdf(pdf_path)
    data = result.to_dict()

    assert isinstance(data, dict)
    assert data["file_path"] == str(pdf_path)
    assert data["total_pages"] == 1
    assert data["has_text"] is True
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 1
    assert "Document metadata test" in data["pages"][0]["text"]


def test_blank_page_pdf(tmp_path: Path):
    """Test processing a PDF that has pages but no extractable text."""
    blank_pdf = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()  # Blank page
    doc.save(str(blank_pdf))
    doc.close()

    result = extract_text_from_pdf(blank_pdf)

    assert result.total_pages == 1
    assert result.has_text is False
    assert result.pages[0].text == ""
    assert result.pages[0].char_count == 0


# -----------------------------------------------------------------------------
# Error Handling Cases
# -----------------------------------------------------------------------------

def test_file_not_found():
    """Test that PDFNotFoundError is raised when the file does not exist."""
    non_existent = Path("non_existent_file.pdf")
    with pytest.raises(PDFNotFoundError, match="does not exist"):
        extract_text_from_pdf(non_existent)


def test_invalid_file_extension(tmp_path: Path):
    """Test that InvalidPDFError is raised for non-PDF extensions."""
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("Plain text content")

    with pytest.raises(InvalidPDFError, match="Only .pdf files are supported"):
        extract_text_from_pdf(txt_file)


def test_corrupted_pdf_file(tmp_path: Path):
    """Test that CorruptedPDFError is raised when reading a broken or invalid PDF file."""
    corrupted_pdf = tmp_path / "corrupted.pdf"
    corrupted_pdf.write_bytes(b"This is not a real PDF binary header")

    with pytest.raises(CorruptedPDFError):
        extract_text_from_pdf(corrupted_pdf)
