"""
VeriDoc — Semantic Retrieval Service (Phase 5)

This module coordinates semantic search by converting user query text into a dense
embedding using the embedding service and querying the FAISSVectorStore for the top-k
most relevant EmbeddedChunks with complete page-level metadata.
"""

from typing import List, Optional, Union
import numpy as np

from backend.services.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    embed_text,
)
from backend.services.vector_store import (
    FAISSVectorStore,
    SearchResult,
)


class RetrievalService:
    """
    Orchestrates semantic retrieval by transforming queries into vector embeddings
    and executing similarity search against a FAISSVectorStore.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        """
        Initialize the RetrievalService.

        Args:
            vector_store: Built FAISSVectorStore instance containing embedded document chunks.
            model_name: Embedding model identifier (defaults to all-MiniLM-L6-v2).
        """
        if not isinstance(vector_store, FAISSVectorStore):
            raise TypeError(
                f"Expected FAISSVectorStore instance, got {type(vector_store).__name__}"
            )
        self.vector_store = vector_store
        self.model_name = model_name

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Converts a user query into an embedding and retrieves the top-k most similar chunks.

        Args:
            query: Natural language question or search query string.
            top_k: Number of most relevant document chunks to return (default: 5).

        Returns:
            List[SearchResult]: Ranked search hits preserving full chunk metadata and similarity scores.

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

        # 3. Generate Query Embedding (L2-normalized)
        query_vector = embed_text(
            clean_query,
            model_name=self.model_name,
            normalize=True,
        )

        # 4. Search FAISS Index
        search_results = self.vector_store.search(
            query_embedding=query_vector,
            top_k=top_k,
        )

        return search_results
