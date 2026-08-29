"""
Tests for Streamlit frontend application integration (Phase 1B & Phase 2B).
"""

from pathlib import Path
try:
    import pymupdf as fitz
except ImportError:
    import fitz  # type: ignore
import pytest
from streamlit.testing.v1 import AppTest

FRONTEND_APP_PATH = Path(__file__).parent.parent / "frontend" / "app.py"


@pytest.fixture
def sample_pdf_bytes():
    """Create in-memory sample PDF bytes with two pages of text."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(fitz.Point(50, 72), "Page 1 sample content for VeriDoc.")
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(50, 72), "Page 2 sample content for VeriDoc.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.fixture
def blank_pdf_bytes():
    """Create in-memory sample PDF bytes with a blank page."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_frontend_initial_render():
    """Test that the initial Streamlit page renders with title and controls."""
    at = AppTest.from_file(str(FRONTEND_APP_PATH)).run()
    assert len(at.title) == 1
    assert "VeriDoc" in at.title[0].value
    assert len(at.sidebar.file_uploader) == 1


def test_frontend_upload_and_chunk_viewer_flow(sample_pdf_bytes):
    """Test full upload, chunking, and visual inspection tabs in the Streamlit UI."""
    at = AppTest.from_file(str(FRONTEND_APP_PATH)).run()

    # Upload PDF document
    at.sidebar.file_uploader[0].upload(filename="research_paper.pdf", content=sample_pdf_bytes)
    at.run()

    # Verify Process Document button appears
    assert len(at.sidebar.button) == 1
    assert "Process Document" in at.sidebar.button[0].label

    # Click Process Document button
    at.sidebar.button[0].click()
    at.run()

    # Verify metrics (Filename, Total Pages, Total Characters, Total Chunks)
    metric_values = [m.value for m in at.metric]
    assert "research_paper.pdf" in metric_values
    assert "2" in metric_values  # 2 pages and 2 chunks

    # Verify expanders created for both Chunks and Pages
    expander_labels = [exp.label for exp in at.expander]
    assert any("Chunk #1" in label for label in expander_labels)
    assert any("Chunk #2" in label for label in expander_labels)
    assert any("Page 1" in label for label in expander_labels)
    assert any("Page 2" in label for label in expander_labels)


def test_frontend_blank_pdf_warning(blank_pdf_bytes):
    """Test that uploading a PDF without extractable text shows the appropriate warning."""
    at = AppTest.from_file(str(FRONTEND_APP_PATH)).run()

    at.sidebar.file_uploader[0].upload(filename="scanned_blank.pdf", content=blank_pdf_bytes)
    at.run()

    at.sidebar.button[0].click()
    at.run()

    assert len(at.warning) >= 1
    assert "No extractable text was found in this PDF." in at.warning[0].value
