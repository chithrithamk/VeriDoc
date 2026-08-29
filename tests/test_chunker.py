"""
Tests for text chunking service with sentence and word boundary preservation (Phase 2).
"""

import pytest

from backend.services.chunker import (
    DocumentChunk,
    chunk_document,
    chunk_text,
)
from backend.services.pdf_processor import ExtractedDocument, PageData


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_extracted_doc():
    """Creates a standard multi-page ExtractedDocument for testing."""
    return ExtractedDocument(
        file_path="data/documents/FINAL_REPORT.pdf",
        total_pages=3,
        pages=[
            PageData(page_number=1, text="Page 1: Overview of AI document intelligence platform.", char_count=54),
            PageData(page_number=2, text="Page 2: Vector retrieval and chunking strategies in RAG.", char_count=56),
            PageData(page_number=3, text="Page 3: Conclusion and evaluation metrics.", char_count=42),
        ],
        total_characters=152,
        has_text=True,
    )


# -----------------------------------------------------------------------------
# Natural Language Boundary Tests
# -----------------------------------------------------------------------------

def test_chunk_does_not_split_words():
    """Test that chunks never cut through the middle of words."""
    text = (
        "The project aims to improve document intelligence and natural language retrieval. "
        "Every single word must remain complete and unbroken across all chunk boundaries."
    )
    # Target chunk_size will land inside a word if arbitrary char splitting is used
    chunks = chunk_text(text, chunk_size=45, chunk_overlap=15)

    assert len(chunks) > 1
    # Check that every chunk starts and ends with whole words (no dangling fragments)
    for chunk in chunks:
        # A whole word boundary means no partial punctuation or broken word prefix/suffix
        words = chunk.split()
        assert len(words) > 0
        # Check that first and last words do not match artificial split segments like "projec" or "t"
        assert "projec" not in words
        assert "intelligenc" not in words


def test_sentence_boundaries_preferred():
    """Test that chunks prefer cutting at punctuation sentence boundaries (. ! ?) over arbitrary words."""
    sentence1 = "The first sentence establishes the primary core architecture."
    sentence2 = "The second sentence introduces vector embeddings and FAISS index."
    sentence3 = "The third sentence describes the Streamlit user interface."
    full_text = f"{sentence1} {sentence2} {sentence3}"

    # Set chunk_size large enough for sentence 1 plus part of sentence 2
    chunk_size = len(sentence1) + 20
    chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=15)

    assert len(chunks) >= 2
    # Chunk 1 should cleanly cut at the end of sentence 1
    assert chunks[0] == sentence1
    assert chunks[0].endswith(".")


def test_word_boundary_fallback_when_no_punctuation():
    """Test that when text has no punctuation, chunks cut cleanly at whitespace/word boundaries."""
    text = "wordone wordtwo wordthree wordfour wordfive wordsix wordseven wordeight wordnine wordten"
    # Choose a chunk_size that lands in the middle of 'wordfour'
    chunk_size = len("wordone wordtwo wordthree wordf")
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=10)

    assert len(chunks) > 1
    # Chunk 1 should end cleanly with 'wordthree'
    assert chunks[0] == "wordone wordtwo wordthree"
    # Next chunk should start cleanly at a word boundary
    assert not chunks[1].startswith("our")
    assert any(w in chunks[1].split() for w in ["wordtwo", "wordthree", "wordfour"])


def test_overlap_starts_at_clean_word_boundary():
    """Test that overlap does not start mid-word in the subsequent chunk."""
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu "
        "xi omicron pi rho sigma tau upsilon phi chi psi omega."
    )
    chunks = chunk_text(text, chunk_size=60, chunk_overlap=20)

    assert len(chunks) >= 2
    for chunk in chunks:
        words = chunk.split()
        for word in words:
            # Each word should be a complete Greek letter name from the source list
            clean_w = word.strip(".,!?")
            assert clean_w in [
                "Alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
                "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"
            ]


# -----------------------------------------------------------------------------
# General Chunking & Metadata Tests
# -----------------------------------------------------------------------------

def test_short_page_produces_one_chunk():
    """Test that a page with text shorter than chunk_size produces exactly one chunk."""
    doc = ExtractedDocument(
        file_path="short_doc.pdf",
        total_pages=1,
        pages=[PageData(page_number=1, text="Short text under chunk size limit.", char_count=35)],
        total_characters=35,
        has_text=True,
    )

    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == 1
    assert chunks[0].page_number == 1
    assert chunks[0].text == "Short text under chunk size limit."
    assert chunks[0].document_name == "short_doc.pdf"


def test_long_page_produces_multiple_chunks():
    """Test that a page with text longer than chunk_size is split into multiple chunks."""
    long_text = "Natural language processing transforms unstructured PDF text into structured chunks. " * 10
    doc = ExtractedDocument(
        file_path="long_doc.pdf",
        total_pages=1,
        pages=[PageData(page_number=1, text=long_text.strip(), char_count=len(long_text.strip()))],
        total_characters=len(long_text.strip()),
        has_text=True,
    )

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200
        assert chunk.page_number == 1
        assert chunk.document_name == "long_doc.pdf"


def test_page_numbers_preserved(sample_extracted_doc):
    """Test that each chunk preserves the page number of its source page."""
    chunks = chunk_document(sample_extracted_doc, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 3
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[2].page_number == 3


def test_document_name_preserved(sample_extracted_doc):
    """Test that document filename is correctly attached to all chunk metadata."""
    chunks = chunk_document(sample_extracted_doc, chunk_size=100, chunk_overlap=20)

    for chunk in chunks:
        assert chunk.document_name == "FINAL_REPORT.pdf"


def test_blank_pages_do_not_create_chunks():
    """Test that empty or blank pages in a document are ignored without creating empty chunks."""
    doc = ExtractedDocument(
        file_path="mixed_doc.pdf",
        total_pages=3,
        pages=[
            PageData(page_number=1, text="Page 1 has valid text.", char_count=22),
            PageData(page_number=2, text="   \n\t  ", char_count=0),  # Blank whitespace page
            PageData(page_number=3, text="Page 3 has valid text.", char_count=22),
        ],
        total_characters=44,
        has_text=True,
    )

    chunks = chunk_document(doc, chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 3
    assert all(len(c.text) > 0 for c in chunks)


def test_empty_document_returns_empty_list():
    """Test that an empty ExtractedDocument returns an empty list of chunks."""
    empty_doc = ExtractedDocument(
        file_path="empty.pdf",
        total_pages=0,
        pages=[],
        total_characters=0,
        has_text=False,
    )

    chunks = chunk_document(empty_doc)
    assert chunks == []


def test_invalid_chunk_size_raises_error():
    """Test that chunk_size <= 0 raises ValueError."""
    doc = ExtractedDocument(file_path="test.pdf", total_pages=1, pages=[PageData(1, "Text", 4)])

    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        chunk_document(doc, chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        chunk_document(doc, chunk_size=-50, chunk_overlap=10)


def test_invalid_overlap_raises_error():
    """Test that negative chunk_overlap raises ValueError."""
    doc = ExtractedDocument(file_path="test.pdf", total_pages=1, pages=[PageData(1, "Text", 4)])

    with pytest.raises(ValueError, match="chunk_overlap must be greater than or equal to 0"):
        chunk_document(doc, chunk_size=500, chunk_overlap=-10)


def test_overlap_greater_or_equal_to_chunk_size_raises_error():
    """Test that chunk_overlap >= chunk_size raises ValueError."""
    doc = ExtractedDocument(file_path="test.pdf", total_pages=1, pages=[PageData(1, "Text", 4)])

    with pytest.raises(ValueError, match="must be strictly smaller than chunk_size"):
        chunk_document(doc, chunk_size=500, chunk_overlap=500)

    with pytest.raises(ValueError, match="must be strictly smaller than chunk_size"):
        chunk_document(doc, chunk_size=500, chunk_overlap=600)


def test_chunk_ids_are_sequential_and_unique():
    """Test that chunk IDs increment sequentially starting from 1 across all pages."""
    doc = ExtractedDocument(
        file_path="report.pdf",
        total_pages=2,
        pages=[
            PageData(page_number=1, text="Text on page one. " * 20, char_count=380),
            PageData(page_number=2, text="Text on page two. " * 20, char_count=380),
        ],
        total_characters=760,
        has_text=True,
    )

    chunks = chunk_document(doc, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 2
    chunk_ids = [c.chunk_id for c in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert chunk_ids == list(range(1, len(chunks) + 1))


def test_chunk_to_dict():
    """Test serialization of DocumentChunk to dictionary."""
    chunk = DocumentChunk(
        chunk_id=1,
        text="Sample chunk content.",
        page_number=2,
        document_name="doc.pdf",
    )

    data = chunk.to_dict()
    assert data["chunk_id"] == 1
    assert data["text"] == "Sample chunk content."
    assert data["page_number"] == 2
    assert data["document_name"] == "doc.pdf"
    assert data["char_count"] == len("Sample chunk content.")
