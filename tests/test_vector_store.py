"""
Tests for FAISS Vector Store service (Phase 4).
"""

import numpy as np
import pytest

from backend.services.embeddings import EmbeddedChunk
from backend.services.vector_store import (
    FAISSVectorStore,
    SearchResult,
)


# -----------------------------------------------------------------------------
# Test Fixtures & Deterministic Vector Helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_embedded_chunks():
    """
    Creates a set of deterministic EmbeddedChunk objects with 4-dimensional unit vectors.
    Vector 1: [1.0, 0.0, 0.0, 0.0] -> Concept A
    Vector 2: [0.0, 1.0, 0.0, 0.0] -> Concept B
    Vector 3: [0.0, 0.0, 1.0, 0.0] -> Concept C
    """
    return [
        EmbeddedChunk(
            chunk_id=1,
            text="Text regarding Document Parsing and Extraction.",
            page_number=1,
            document_name="architecture_doc.pdf",
            embedding=[1.0, 0.0, 0.0, 0.0],
            char_count=46,
        ),
        EmbeddedChunk(
            chunk_id=2,
            text="Text regarding Text Chunking and Tokenization.",
            page_number=2,
            document_name="architecture_doc.pdf",
            embedding=[0.0, 1.0, 0.0, 0.0],
            char_count=46,
        ),
        EmbeddedChunk(
            chunk_id=3,
            text="Text regarding FAISS Vector Indexing.",
            page_number=3,
            document_name="architecture_doc.pdf",
            embedding=[0.0, 0.0, 1.0, 0.0],
            char_count=37,
        ),
    ]


# -----------------------------------------------------------------------------
# Unit Tests for FAISSVectorStore Initialization & Building
# -----------------------------------------------------------------------------

def test_empty_store_initialization():
    """Test that an empty vector store can be initialized with default state."""
    store = FAISSVectorStore()
    assert store.is_built() is False
    assert len(store) == 0
    assert store.index is None
    assert store.dimension is None


def test_build_index_from_chunks(sample_embedded_chunks):
    """Test that building the FAISS index registers all vectors and dimensions."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)

    assert store.is_built() is True
    assert len(store) == 3
    assert store.dimension == 4


def test_empty_chunk_list_build():
    """Test that passing an empty list to build() is handled safely."""
    store = FAISSVectorStore()
    store.build([])

    assert store.is_built() is False
    assert len(store) == 0


def test_dynamic_embedding_dimensions():
    """Test that the vector store dynamically accepts different vector dimensions (e.g. 128-dim)."""
    dim = 128
    vec = [0.0] * dim
    vec[0] = 1.0
    chunk = EmbeddedChunk(
        chunk_id=1,
        text="High-dim vector chunk.",
        page_number=1,
        document_name="test.pdf",
        embedding=vec,
    )

    store = FAISSVectorStore()
    store.build([chunk])

    assert store.dimension == 128
    assert len(store) == 1


# -----------------------------------------------------------------------------
# Unit Tests for Search & Retrieval
# -----------------------------------------------------------------------------

def test_search_returns_exact_nearest_match(sample_embedded_chunks):
    """Test that querying with an exact vector returns the corresponding chunk with score ~1.0."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)

    # Query matching Vector 3 ([0.0, 0.0, 1.0, 0.0]) -> Chunk #3
    query = [0.0, 0.0, 1.0, 0.0]
    results = store.search(query, top_k=1)

    assert len(results) == 1
    top_hit = results[0]
    assert isinstance(top_hit, SearchResult)
    assert top_hit.chunk.chunk_id == 3
    assert top_hit.chunk.page_number == 3
    assert top_hit.chunk.text == "Text regarding FAISS Vector Indexing."
    assert pytest.approx(top_hit.score, 0.001) == 1.0


def test_search_results_ordered_by_similarity_score(sample_embedded_chunks):
    """Test that multiple search results are sorted in descending order of similarity score."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)

    # Query closer to Vector 1 (0.8) and slightly to Vector 2 (0.6)
    query = [0.8, 0.6, 0.0, 0.0]
    results = store.search(query, top_k=3)

    assert len(results) == 3
    # First should be chunk 1, second chunk 2, third chunk 3
    assert results[0].chunk.chunk_id == 1
    assert results[1].chunk.chunk_id == 2
    assert results[2].chunk.chunk_id == 3

    # Scores must be strictly non-increasing
    scores = [r.score for r in results]
    assert scores[0] >= scores[1] >= scores[2]


def test_top_k_limiting(sample_embedded_chunks):
    """Test that top_k restricts the number of returned SearchResult items."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)

    results_k1 = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    results_k2 = store.search([1.0, 0.0, 0.0, 0.0], top_k=2)

    assert len(results_k1) == 1
    assert len(results_k2) == 2


def test_top_k_larger_than_store_size(sample_embedded_chunks):
    """Test that top_k greater than store size returns only available chunks without error."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=100)
    assert len(results) == 3


def test_metadata_preservation_after_search(sample_embedded_chunks):
    """Test that all original metadata is preserved on the retrieved EmbeddedChunk."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)

    results = store.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    chunk = results[0].chunk

    assert chunk.chunk_id == 1
    assert chunk.page_number == 1
    assert chunk.document_name == "architecture_doc.pdf"
    assert chunk.char_count == 46
    assert chunk.embedding == [1.0, 0.0, 0.0, 0.0]


# -----------------------------------------------------------------------------
# Unit Tests for Error Handling & Edge Cases
# -----------------------------------------------------------------------------

def test_search_before_build_raises_error():
    """Test that searching an unbuilt vector store raises RuntimeError."""
    store = FAISSVectorStore()
    with pytest.raises(RuntimeError, match="has not been built yet"):
        store.search([1.0, 0.0, 0.0, 0.0], top_k=1)


def test_invalid_top_k_raises_error(sample_embedded_chunks):
    """Test that top_k <= 0 raises ValueError."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        store.search([1.0, 0.0, 0.0, 0.0], top_k=0)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        store.search([1.0, 0.0, 0.0, 0.0], top_k=-5)


def test_dimension_mismatch_raises_error(sample_embedded_chunks):
    """Test that querying with a vector of mismatched dimension raises ValueError."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)  # dim = 4

    with pytest.raises(ValueError, match="does not match index dimension"):
        store.search([1.0, 0.0, 0.0], top_k=1)  # 3 dims instead of 4


def test_search_result_to_dict():
    """Test serialization of SearchResult to dictionary format."""
    chunk = EmbeddedChunk(
        chunk_id=1,
        text="Content",
        page_number=2,
        document_name="file.pdf",
        embedding=[0.5, 0.5],
    )
    res = SearchResult(chunk=chunk, score=0.92)
    data = res.to_dict()

    assert data["score"] == 0.92
    assert data["chunk"]["chunk_id"] == 1
    assert data["chunk"]["page_number"] == 2
    assert data["chunk"]["document_name"] == "file.pdf"


def test_clear_store(sample_embedded_chunks):
    """Test that clearing the store resets internal index and chunks."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)
    assert len(store) == 3

    store.clear()
    assert store.is_built() is False
    assert len(store) == 0
