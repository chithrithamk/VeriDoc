"""
Tests for text embedding service (Phase 3).
"""

import numpy as np
import pytest

from backend.services.chunker import DocumentChunk
from backend.services.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddedChunk,
    embed_chunks,
    embed_text,
    embed_texts,
    get_embedding_dimension,
    load_embedding_model,
)


# -----------------------------------------------------------------------------
# Unit Tests for Model Loading & Dimensions
# -----------------------------------------------------------------------------

def test_model_loading():
    """Test that the embedding model can be loaded and cached successfully."""
    model = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
    assert model is not None

    # Verify model is cached and returns the same instance
    model_cached = load_embedding_model(DEFAULT_EMBEDDING_MODEL)
    assert model is model_cached


def test_embedding_dimension():
    """Test that the model produces the expected consistent vector dimensionality (384)."""
    dim = get_embedding_dimension(DEFAULT_EMBEDDING_MODEL)
    assert isinstance(dim, int)
    assert dim == 384


# -----------------------------------------------------------------------------
# Unit Tests for embed_text
# -----------------------------------------------------------------------------

def test_embed_single_text():
    """Test that a standard text string produces a numerical embedding vector."""
    text = "VeriDoc is an AI document intelligence platform."
    vector = embed_text(text)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (384,)
    assert vector.dtype == np.float32
    assert all(isinstance(val, (float, np.floating)) for val in vector)


def test_different_texts_produce_same_dimension():
    """Test that different texts produce vectors with the exact same dimension."""
    vec1 = embed_text("Short text.")
    vec2 = embed_text("A considerably longer paragraph explaining retrieval-augmented generation and vector databases.")

    assert vec1.shape == vec2.shape
    assert vec1.shape == (384,)


def test_empty_and_whitespace_text_handling():
    """Test that empty or whitespace strings are handled safely without crashing."""
    vec_empty = embed_text("")
    vec_whitespace = embed_text("   \n\t  ")

    assert vec_empty.shape == (384,)
    assert vec_whitespace.shape == (384,)


# -----------------------------------------------------------------------------
# Unit Tests for embed_texts (Batch)
# -----------------------------------------------------------------------------

def test_embed_texts_batch():
    """Test batch embedding generation for multiple text strings."""
    texts = [
        "First sentence for embedding.",
        "Second document sentence.",
        "Third distinct sentence about machine learning.",
    ]
    matrix = embed_texts(texts, batch_size=2)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 384)
    assert matrix.dtype == np.float32


def test_embed_texts_empty_list():
    """Test that passing an empty list of texts returns an empty matrix of shape (0, 384)."""
    matrix = embed_texts([])
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (0, 384)


# -----------------------------------------------------------------------------
# Unit Tests for embed_chunks
# -----------------------------------------------------------------------------

def test_embed_chunks_preserves_metadata_and_order():
    """Test that embed_chunks associates embeddings with chunks and preserves 1-to-1 ordering."""
    chunks = [
        DocumentChunk(chunk_id=1, text="Chunk one text about PDF parsing.", page_number=1, document_name="doc1.pdf"),
        DocumentChunk(chunk_id=2, text="Chunk two text about vector embeddings.", page_number=2, document_name="doc1.pdf"),
        DocumentChunk(chunk_id=3, text="Chunk three text about similarity retrieval.", page_number=3, document_name="doc1.pdf"),
    ]

    embedded = embed_chunks(chunks)

    assert len(embedded) == 3

    # Check 1-to-1 ordering and metadata preservation
    for i, (orig, emb) in enumerate(zip(chunks, embedded)):
        assert isinstance(emb, EmbeddedChunk)
        assert emb.chunk_id == orig.chunk_id
        assert emb.text == orig.text
        assert emb.page_number == orig.page_number
        assert emb.document_name == orig.document_name
        assert emb.char_count == len(orig.text)
        assert isinstance(emb.embedding, list)
        assert len(emb.embedding) == 384


def test_embed_chunks_empty_list():
    """Test that passing an empty chunk list returns an empty list."""
    embedded = embed_chunks([])
    assert embedded == []


def test_embedded_chunk_to_dict():
    """Test dictionary serialization of EmbeddedChunk."""
    chunk = EmbeddedChunk(
        chunk_id=10,
        text="Sample text content.",
        page_number=4,
        document_name="report.pdf",
        embedding=[0.1] * 384,
    )
    data = chunk.to_dict()

    assert data["chunk_id"] == 10
    assert data["text"] == "Sample text content."
    assert data["page_number"] == 4
    assert data["document_name"] == "report.pdf"
    assert data["char_count"] == len("Sample text content.")
    assert len(data["embedding"]) == 384


# -----------------------------------------------------------------------------
# Semantic Similarity Test
# -----------------------------------------------------------------------------

def test_semantic_similarity_relative_ordering():
    """
    Test that semantically related texts have higher cosine similarity than unrelated texts.
    With L2-normalized embeddings, cosine similarity equals the dot product (np.dot).
    """
    query = "artificial intelligence and machine learning"
    related_text = "deep learning neural networks and AI algorithms"
    unrelated_text = "how to bake chocolate chip cookies in an oven"

    q_vec = embed_text(query, normalize=True)
    rel_vec = embed_text(related_text, normalize=True)
    unrel_vec = embed_text(unrelated_text, normalize=True)

    # Dot product of normalized vectors = Cosine Similarity in [-1, 1]
    sim_related = float(np.dot(q_vec, rel_vec))
    sim_unrelated = float(np.dot(q_vec, unrel_vec))

    assert sim_related > sim_unrelated
    assert sim_related > 0.4  # Strong positive semantic correlation
