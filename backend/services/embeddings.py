"""
VeriDoc — Text Embeddings Service (Phase 3)

This module handles semantic vector embedding generation using Sentence Transformers.
It converts raw text strings and DocumentChunk instances into normalized numerical
vectors suitable for vector databases (FAISS) and similarity search.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.services.chunker import DocumentChunk


# Default asymmetric embedding model (fast, local, and CPU-friendly)
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-small-v2"

# Global cache for loaded model instances to prevent reloading overhead
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class EmbeddedChunk:
    """
    Represents a DocumentChunk paired with its dense vector embedding.
    """
    chunk_id: int
    text: str
    page_number: int
    document_name: str
    embedding: List[float]
    char_count: int = 0

    def __post_init__(self):
        if self.char_count == 0 and self.text:
            self.char_count = len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert embedded chunk to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_number": self.page_number,
            "document_name": self.document_name,
            "char_count": self.char_count,
            "embedding": self.embedding,
        }


# -----------------------------------------------------------------------------
# Model Management & Introspection
# -----------------------------------------------------------------------------

def _get_model_prefix(model_name: str, is_query: bool) -> str:
    """
    Returns the required prefix for asymmetric embedding models:
    - E5 models ('intfloat/e5-*'): 'query: ' for questions, 'passage: ' for documents.
    - BGE models ('BAAI/bge-*'): 'Represent this sentence for searching relevant passages: ' for queries.
    - Symmetric models: '' (no prefix).
    """
    name_lower = model_name.lower()
    if "e5" in name_lower:
        return "query: " if is_query else "passage: "
    elif "bge" in name_lower:
        return "Represent this sentence for searching relevant passages: " if is_query else ""
    return ""


def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> SentenceTransformer:
    """
    Loads or retrieves a cached SentenceTransformer model instance.

    Args:
        model_name: HuggingFace model identifier or local directory path.

    Returns:
        SentenceTransformer: Initialized model instance ready for inference.
    """
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def get_embedding_dimension(model_name: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """
    Returns the vector dimension produced by the specified model (e.g. 384 for e5-small-v2).
    """
    model = load_embedding_model(model_name)
    if hasattr(model, "get_embedding_dimension"):
        dim = model.get_embedding_dimension()
    else:
        dim = model.get_sentence_embedding_dimension()
    return int(dim) if dim is not None else 384


# -----------------------------------------------------------------------------
# Embedding Generation Functions
# -----------------------------------------------------------------------------

def embed_text(
    text: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize: bool = True,
    is_query: bool = False,
) -> np.ndarray:
    """
    Generates a dense vector embedding for an individual text string.

    Args:
        text: Input string to embed.
        model_name: Model identifier to use.
        normalize: Whether to L2-normalize the output vector (default: True).
        is_query: Whether the text is a search query (applies asymmetric query prefix).

    Returns:
        np.ndarray: 1D numerical embedding array (dtype float32).

    Raises:
        TypeError: If input text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}")

    model = load_embedding_model(model_name)

    # Handle empty/whitespace strings safely without crashing
    if not text.strip():
        dim = get_embedding_dimension(model_name)
        return np.zeros(dim, dtype=np.float32)

    prefix = _get_model_prefix(model_name, is_query=is_query)
    prefixed_text = f"{prefix}{text}" if prefix else text

    embedding = model.encode(prefixed_text, normalize_embeddings=normalize, show_progress_bar=False)
    return np.asarray(embedding, dtype=np.float32)


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize: bool = True,
    batch_size: int = 32,
    is_query: bool = False,
) -> np.ndarray:
    """
    Generates dense vector embeddings for a list of text strings in batches.

    Args:
        texts: List of text strings to encode.
        model_name: Model identifier to use.
        normalize: Whether to L2-normalize output vectors (default: True).
        batch_size: Batch size for model inference (default: 32).
        is_query: Whether the texts are search queries (applies query prefix).

    Returns:
        np.ndarray: 2D numerical embedding matrix of shape (N, dimension).

    Raises:
        TypeError: If input texts is not a list.
    """
    if not isinstance(texts, list):
        raise TypeError(f"Input texts must be a list of strings, got {type(texts).__name__}")

    if len(texts) == 0:
        dim = get_embedding_dimension(model_name)
        return np.empty((0, dim), dtype=np.float32)

    model = load_embedding_model(model_name)
    prefix = _get_model_prefix(model_name, is_query=is_query)

    if prefix:
        prefixed_texts = [f"{prefix}{t}" if t.strip() else t for t in texts]
    else:
        prefixed_texts = texts

    embeddings = model.encode(
        prefixed_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_chunks(
    chunks: List[DocumentChunk],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    normalize: bool = True,
    batch_size: int = 32,
) -> List[EmbeddedChunk]:
    """
    Generates embeddings for a list of DocumentChunks in a batch operation,
    preserving metadata and strict 1-to-1 ordering. Chunks are embedded with
    passage prefixing.

    Args:
        chunks: List of DocumentChunk instances.
        model_name: Model identifier to use.
        normalize: Whether to L2-normalize output vectors (default: True).
        batch_size: Batch size for model inference (default: 32).

    Returns:
        List[EmbeddedChunk]: List of EmbeddedChunk instances with embeddings attached.

    Raises:
        TypeError: If input chunks is not a list.
    """
    if not isinstance(chunks, list):
        raise TypeError(f"Expected a list of DocumentChunk instances, got {type(chunks).__name__}")

    if len(chunks) == 0:
        return []

    # Extract chunk texts preserving index sequence
    texts = [chunk.text for chunk in chunks]

    # Batch encode texts as passages (is_query=False)
    embeddings_matrix = embed_texts(
        texts,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
        is_query=False,
    )

    # Attach embeddings back to chunk metadata (preserving raw chunk.text for citations)
    embedded_chunks: List[EmbeddedChunk] = []
    for chunk, vector in zip(chunks, embeddings_matrix):
        embedded_chunk = EmbeddedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            page_number=chunk.page_number,
            document_name=chunk.document_name,
            embedding=vector.tolist(),
            char_count=chunk.char_count,
        )
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks
