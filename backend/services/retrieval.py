"""
VeriDoc — Semantic Retrieval Service (Phase 5)

This module coordinates semantic search by converting user query text into a dense
embedding using the embedding service and querying the FAISSVectorStore for the top-k
most relevant EmbeddedChunks with complete page-level metadata.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sentence_transformers import CrossEncoder

from backend.services.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    embed_text,
)
from backend.services.vector_store import (
    FAISSVectorStore,
    SearchResult,
)

# Default lightweight cross-encoder model for passage reranking
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global cache for loaded CrossEncoder instances
_RERANKER_CACHE: Dict[str, CrossEncoder] = {}


def load_reranker_model(model_name: str = DEFAULT_RERANKER_MODEL) -> CrossEncoder:
    """
    Loads or retrieves a cached CrossEncoder model instance.

    Args:
        model_name: HuggingFace cross-encoder model identifier.

    Returns:
        CrossEncoder: Initialized CrossEncoder model instance ready for scoring.
    """
    if model_name not in _RERANKER_CACHE:
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


class RetrievalService:
    """
    Orchestrates semantic retrieval by transforming queries into vector embeddings,
    fetching an initial candidate pool from FAISS, and performing precision cross-encoder
    reranking.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        use_reranker: bool = True,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        reranker_instance: Optional[Any] = None,
    ):
        """
        Initialize the RetrievalService.

        Args:
            vector_store: Built FAISSVectorStore instance containing embedded document chunks.
            model_name: Embedding model identifier (defaults to intfloat/e5-small-v2).
            use_reranker: Whether to apply CrossEncoder reranking on candidate chunks (default: True).
            reranker_model: CrossEncoder model identifier (defaults to ms-marco-MiniLM-L-6-v2).
            reranker_instance: Optional pre-loaded CrossEncoder instance (useful for mocking/testing).
        """
        if not isinstance(vector_store, FAISSVectorStore):
            raise TypeError(
                f"Expected FAISSVectorStore instance, got {type(vector_store).__name__}"
            )
        self.vector_store = vector_store
        self.model_name = model_name
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self._reranker_instance = reranker_instance

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Converts a user query into an asymmetric query embedding, retrieves candidate chunks
        from FAISS, and optionally reranks them via CrossEncoder.

        Args:
            query: Natural language question or search query string.
            top_k: Number of most relevant document chunks to return (default: 5).

        Returns:
            List[SearchResult]: Ranked search hits preserving full chunk metadata and similarity/rerank scores.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty/whitespace or top_k <= 0.
            RuntimeError: If vector store has not been built yet.
        """
        # 1. Query Validation
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}")

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query string cannot be empty or whitespace-only.")

        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")

        # 2. Vector Store State Validation
        if not self.vector_store.is_built():
            if not getattr(self.vector_store, "_has_called_build", False):
                raise RuntimeError(
                    "Vector store has not been built yet. Please build the FAISS index before retrieving."
                )
            return []

        if len(self.vector_store) == 0:
            return []

        # 3. Generate Asymmetric Query Embedding (L2-normalized with query prefix)
        query_vector = embed_text(
            clean_query,
            model_name=self.model_name,
            normalize=True,
            is_query=True,
        )

        # 4. Stage 1: Search FAISS Index (fetch wider candidate pool when reranking)
        candidate_k = max(top_k * 4, 15) if self.use_reranker else top_k
        candidates = self.vector_store.search(
            query_embedding=query_vector,
            top_k=candidate_k,
        )

        if not candidates:
            return []

        # 5. Stage 2: Cross-Encoder Precision Reranking
        if self.use_reranker and len(candidates) > 0:
            reranker = self._reranker_instance or load_reranker_model(self.reranker_model)
            pairs = [[clean_query, res.chunk.text] for res in candidates]
            rerank_scores = reranker.predict(pairs)

            reranked_results: List[SearchResult] = []
            for res, score in zip(candidates, rerank_scores):
                reranked_results.append(
                    SearchResult(
                        chunk=res.chunk,
                        score=float(score),
                    )
                )

            # Sort descending by CrossEncoder relevance score
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            return reranked_results[:top_k]

        return candidates[:top_k]
