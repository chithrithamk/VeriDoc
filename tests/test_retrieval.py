"""
Tests for semantic retrieval service (Phase 5).
"""

from unittest.mock import patch
import numpy as np
import pytest

from backend.services.embeddings import EmbeddedChunk
from backend.services.retrieval import RetrievalService
from backend.services.vector_store import FAISSVectorStore, SearchResult


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_embedded_chunks():
    """Deterministic EmbeddedChunk list with 4D vectors."""
    return [
        EmbeddedChunk(
            chunk_id=1,
            text="The project architecture utilizes PyMuPDF for PDF text extraction.",
            page_number=1,
            document_name="architecture.pdf",
            embedding=[1.0, 0.0, 0.0, 0.0],
            char_count=69,
        ),
        EmbeddedChunk(
            chunk_id=2,
            text="Text chunking splits document content into overlapping segments.",
            page_number=2,
            document_name="architecture.pdf",
            embedding=[0.0, 1.0, 0.0, 0.0],
            char_count=65,
        ),
        EmbeddedChunk(
            chunk_id=3,
            text="FAISS vector store provides efficient similarity search over embeddings.",
            page_number=3,
            document_name="architecture.pdf",
            embedding=[0.0, 0.0, 1.0, 0.0],
            char_count=73,
        ),
    ]


@pytest.fixture
def built_vector_store(sample_embedded_chunks):
    """Returns a FAISSVectorStore already populated with sample chunks."""
    store = FAISSVectorStore()
    store.build(sample_embedded_chunks)
    return store


# -----------------------------------------------------------------------------
# Unit Tests for RetrievalService
# -----------------------------------------------------------------------------

def test_valid_query_returns_results(built_vector_store):
    """Test that a valid query invokes embedding and returns SearchResult list."""
    service = RetrievalService(vector_store=built_vector_store)

    # Mock embed_text to return a vector matching chunk 3
    with patch("backend.services.retrieval.embed_text", return_value=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)) as mock_embed:
        results = service.retrieve("How does FAISS search work?", top_k=2)

        mock_embed.assert_called_once_with(
            "How does FAISS search work?",
            model_name=service.model_name,
            normalize=True,
        )

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].chunk.chunk_id == 3
        assert results[0].chunk.page_number == 3
        assert "FAISS vector store" in results[0].chunk.text
        assert pytest.approx(results[0].score, 0.001) == 1.0


def test_top_k_limiting(built_vector_store):
    """Test that top_k restricts the number of returned chunks."""
    service = RetrievalService(vector_store=built_vector_store)

    with patch("backend.services.retrieval.embed_text", return_value=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)):
        results_1 = service.retrieve("Query text", top_k=1)
        results_2 = service.retrieve("Query text", top_k=2)

        assert len(results_1) == 1
        assert len(results_2) == 2


def test_results_ordered_by_similarity(built_vector_store):
    """Test that returned search results are ordered by similarity score in descending order."""
    service = RetrievalService(vector_store=built_vector_store)

    # Vector closer to chunk 2 (0.9) than chunk 1 (0.4)
    with patch("backend.services.retrieval.embed_text", return_value=np.array([0.4, 0.9, 0.0, 0.0], dtype=np.float32)):
        results = service.retrieve("Chunking query", top_k=3)

        assert len(results) == 3
        assert results[0].chunk.chunk_id == 2
        assert results[1].chunk.chunk_id == 1
        assert results[2].chunk.chunk_id == 3
        assert results[0].score >= results[1].score >= results[2].score


def test_metadata_and_citations_preserved(built_vector_store):
    """Test that chunk ID, page number, document name, text, and embedding are preserved."""
    service = RetrievalService(vector_store=built_vector_store)

    with patch("backend.services.retrieval.embed_text", return_value=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)):
        results = service.retrieve("PDF extraction query", top_k=1)

        assert len(results) == 1
        chunk = results[0].chunk
        assert chunk.chunk_id == 1
        assert chunk.page_number == 1
        assert chunk.document_name == "architecture.pdf"
        assert chunk.char_count == 69
        assert chunk.embedding == [1.0, 0.0, 0.0, 0.0]


# -----------------------------------------------------------------------------
# Validation & Error Handling Tests
# -----------------------------------------------------------------------------

def test_empty_query_raises_value_error(built_vector_store):
    """Test that an empty query string raises ValueError."""
    service = RetrievalService(vector_store=built_vector_store)
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        service.retrieve("")


def test_whitespace_only_query_raises_value_error(built_vector_store):
    """Test that a whitespace-only query raises ValueError."""
    service = RetrievalService(vector_store=built_vector_store)
    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        service.retrieve("   \n\t  ")


def test_non_string_query_raises_type_error(built_vector_store):
    """Test that a non-string query raises TypeError."""
    service = RetrievalService(vector_store=built_vector_store)
    with pytest.raises(TypeError, match="Query must be a string"):
        service.retrieve(12345)  # type: ignore


def test_invalid_top_k_raises_value_error(built_vector_store):
    """Test that top_k <= 0 raises ValueError."""
    service = RetrievalService(vector_store=built_vector_store)
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        service.retrieve("Valid query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        service.retrieve("Valid query", top_k=-3)


def test_retrieval_from_unbuilt_store_raises_error():
    """Test that calling retrieve on an unbuilt store raises RuntimeError."""
    empty_store = FAISSVectorStore()
    service = RetrievalService(vector_store=empty_store)

    with pytest.raises(RuntimeError, match="has not been built yet"):
        service.retrieve("Valid query", top_k=1)


def test_retrieval_from_empty_store():
    """Test that retrieval from a store built with [] returns an empty list."""
    store = FAISSVectorStore()
    store.build([])
    service = RetrievalService(vector_store=store)

    # Empty store (not initialized with vectors) returns empty list
    results = service.retrieve("Query", top_k=5)
    assert results == []


# -----------------------------------------------------------------------------
# End-to-End Integration Test
# -----------------------------------------------------------------------------

def test_retrieval_integration_with_real_embeddings():
    """
    Integration test using the real embedding model:
    Question -> embed_text -> FAISS search -> relevant chunks.
    """
    from backend.services.embeddings import embed_chunks
    from backend.services.chunker import DocumentChunk

    chunks = [
        DocumentChunk(
            chunk_id=1,
            text="The capital of France is Paris, famous for the Eiffel Tower.",
            page_number=1,
            document_name="geography.pdf",
        ),
        DocumentChunk(
            chunk_id=2,
            text="Photosynthesis is the biological process by which green plants create food from sunlight.",
            page_number=2,
            document_name="biology.pdf",
        ),
        DocumentChunk(
            chunk_id=3,
            text="Quantum computing utilizes qubits, superposition, and entanglement for calculation.",
            page_number=3,
            document_name="physics.pdf",
        ),
    ]

    # 1. Generate real embeddings
    embedded_chunks = embed_chunks(chunks)

    # 2. Build FAISS vector store
    store = FAISSVectorStore()
    store.build(embedded_chunks)

    # 3. Initialize retrieval service
    retrieval_service = RetrievalService(vector_store=store)

    # 4. Search for botany / biology question
    results = retrieval_service.retrieve("How do plants generate energy from sunlight?", top_k=1)

    assert len(results) == 1
    top_chunk = results[0].chunk
    assert top_chunk.chunk_id == 2
    assert top_chunk.page_number == 2
    assert top_chunk.document_name == "biology.pdf"
    assert "Photosynthesis" in top_chunk.text
    assert results[0].score > 0.4  # Strong semantic match
