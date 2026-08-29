"""
VeriDoc — FAISS Vector Store Service (Phase 4)

This module implements a reusable in-memory FAISS vector index using IndexFlatIP
(Inner Product / Cosine Similarity for normalized embeddings) and maintains strict
1-to-1 mapping back to original EmbeddedChunk metadata for transparent citations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import faiss
import numpy as np

from backend.services.embeddings import EmbeddedChunk


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class SearchResult:
    """
    Represents a single search hit containing the matched EmbeddedChunk and its similarity score.
    """
    chunk: EmbeddedChunk
    score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert SearchResult to dictionary format."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
        }


# -----------------------------------------------------------------------------
# FAISS Vector Store
# -----------------------------------------------------------------------------

class FAISSVectorStore:
    """
    In-memory vector store powered by FAISS IndexFlatIP.
    Stores dense chunk embeddings and maintains integer position mappings back to
    the original EmbeddedChunk objects.
    """

    def __init__(self):
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dimension: Optional[int] = None
        self.chunks: List[EmbeddedChunk] = []
        self._has_called_build: bool = False

    def is_built(self) -> bool:
        """Returns True if the index contains indexed vectors."""
        return self.index is not None and len(self.chunks) > 0

    def __len__(self) -> int:
        """Returns the number of indexed vectors/chunks."""
        return len(self.chunks)

    def build(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        """
        Builds a FAISS IndexFlatIP index from a list of EmbeddedChunk objects.

        Args:
            embedded_chunks: List of EmbeddedChunk instances containing vector embeddings.

        Raises:
            TypeError: If input is not a list.
            ValueError: If embeddings have inconsistent dimensions.
        """
        if not isinstance(embedded_chunks, list):
            raise TypeError(
                f"Expected a list of EmbeddedChunk instances, got {type(embedded_chunks).__name__}"
            )

        self._has_called_build = True

        # Handle empty input list safely
        if len(embedded_chunks) == 0:
            self.index = None
            self.dimension = None
            self.chunks = []
            return

        # Determine dimensionality dynamically from the first chunk
        first_embedding = embedded_chunks[0].embedding
        dimension = len(first_embedding)
        if dimension == 0:
            raise ValueError("EmbeddedChunk embedding vector cannot be empty.")

        # Convert all embeddings into a 2D float32 numpy array
        vectors = np.array([c.embedding for c in embedded_chunks], dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape[1] != dimension:
            raise ValueError(
                f"Inconsistent vector shapes: expected 2D array of dimension {dimension}, "
                f"got shape {vectors.shape}"
            )

        # Create FAISS IndexFlatIP (Inner Product)
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)

        # Store internal state and 1-to-1 chunk mappings
        self.index = index
        self.dimension = dimension
        self.chunks = list(embedded_chunks)

    def search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Searches the FAISS index for the top_k most similar chunks.

        Args:
            query_embedding: Query vector as a 1D/2D float array or list of floats.
            top_k: Number of most similar results to return (default: 5).

        Returns:
            List[SearchResult]: Ordered list of search results with chunks and similarity scores.

        Raises:
            ValueError: If top_k <= 0 or query embedding dimension mismatches.
            RuntimeError: If search is called before building the index.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

        if not self._has_called_build:
            raise RuntimeError(
                "FAISS vector store has not been built yet. Call build() with EmbeddedChunk objects first."
            )

        if len(self.chunks) == 0 or self.index is None:
            return []

        # Format and validate query vector
        query_vec = np.asarray(query_embedding, dtype=np.float32)

        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)

        if query_vec.ndim != 2 or query_vec.shape[1] != self.dimension:
            raise ValueError(
                f"Query vector dimension {query_vec.shape[1]} does not match "
                f"index dimension {self.dimension}"
            )

        # Clamp top_k to the number of indexed chunks
        effective_k = min(top_k, len(self.chunks))
        if effective_k == 0:
            return []

        # Perform FAISS similarity search
        distances, indices = self.index.search(query_vec, effective_k)

        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS sentinel for no match / unpopulated slots
                continue
            matched_chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    chunk=matched_chunk,
                    score=float(score),
                )
            )

        return results

    def clear(self) -> None:
        """Resets the vector index and clears all stored chunks."""
        self.index = None
        self.dimension = None
        self.chunks = []
        self._has_called_build = False
