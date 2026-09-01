"""
VeriDoc — Full RAG Pipeline Service (Phase 7)

This module provides an end-to-end RAG pipeline orchestrator that integrates:
1. PDF Text Extraction (PyMuPDF)
2. Text Chunking (Boundary-aware Chunker)
3. Embedding Generation (Sentence Transformers)
4. FAISS Vector Store Indexing (IndexFlatIP)
5. Semantic Retrieval (Top-k Similarity Search)
6. Grounded Answer Generation (Google Gemini)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.services.chunker import DocumentChunk, chunk_document
from backend.services.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddedChunk,
    embed_chunks,
)
from backend.services.generator import (
    DEFAULT_GEMINI_MODEL,
    AnswerGenerator,
    GeneratedAnswer,
    LLMConfigurationError,
    LLMGenerationError,
    generate_rag_answer,
)
from backend.services.pdf_processor import (
    ExtractedDocument,
    extract_text_from_pdf,
)
from backend.services.retrieval import RetrievalService
from backend.services.vector_store import FAISSVectorStore, SearchResult


class RAGPipeline:
    """
    End-to-end RAG Pipeline orchestrator for document ingestion and question answering.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: str = DEFAULT_GEMINI_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the RAG pipeline with vector storage, retrieval, and generation services.

        Args:
            api_key: Optional Gemini API key. If omitted, resolved from environment.
            llm_model: Target Gemini model name.
            embedding_model: Local sentence-transformers model name.
            llm_client: Optional pre-configured LLM client (e.g. for testing/mocking).
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Vector Store & Retrieval Service
        self.vector_store = FAISSVectorStore()
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            model_name=self.embedding_model,
        )

        # Answer Generator Service (lazy or direct init)
        self.api_key = api_key
        self.llm_client = llm_client
        self._generator: Optional[AnswerGenerator] = None

        # Document & Chunk State
        self.document: Optional[ExtractedDocument] = None
        self.chunks: List[DocumentChunk] = []
        self.embedded_chunks: List[EmbeddedChunk] = []

    @property
    def generator(self) -> AnswerGenerator:
        """Lazily initialize and return the AnswerGenerator instance."""
        if self._generator is None:
            self._generator = AnswerGenerator(
                api_key=self.api_key,
                model_name=self.llm_model,
                client=self.llm_client,
            )
        return self._generator

    def is_ready(self) -> bool:
        """Returns True if the pipeline has an active, populated vector index."""
        return self.vector_store.is_built() and len(self.vector_store) > 0

    def get_stats(self) -> Dict[str, Any]:
        """Returns summary statistics for the currently ingested document."""
        if not self.document:
            return {
                "document_name": None,
                "total_pages": 0,
                "total_characters": 0,
                "total_chunks": 0,
                "indexed_vectors": 0,
                "is_ready": False,
            }
        return {
            "document_name": Path(self.document.file_path).name,
            "total_pages": self.document.total_pages,
            "total_characters": self.document.total_characters,
            "total_chunks": len(self.chunks),
            "indexed_vectors": len(self.vector_store),
            "is_ready": self.is_ready(),
        }

    def ingest_pdf(
        self,
        pdf_path: Union[str, Path],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Dict[str, Any]:
        """
        Full ingestion pipeline: Extract -> Chunk -> Embed -> Index.

        Args:
            pdf_path: Path to the target PDF file.
            chunk_size: Target character count per chunk.
            chunk_overlap: Overlapping characters between consecutive chunks.

        Returns:
            Dict[str, Any]: Ingestion summary and statistics.
        """
        # 1. Extract text
        extracted_doc = extract_text_from_pdf(pdf_path)

        # 2. Ingest structured document
        return self.ingest_document(
            extracted_doc=extracted_doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_document(
        self,
        extracted_doc: ExtractedDocument,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> Dict[str, Any]:
        """
        Ingests an already extracted ExtractedDocument: Chunk -> Embed -> Index.

        Args:
            extracted_doc: ExtractedDocument instance from pdf_processor.
            chunk_size: Target character count per chunk.
            chunk_overlap: Overlapping characters between chunks.

        Returns:
            Dict[str, Any]: Summary dictionary with ingestion stats.
        """
        self.clear()
        self.document = extracted_doc

        # 1. Chunk document
        self.chunks = chunk_document(
            document=extracted_doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not self.chunks:
            self.vector_store.build([])
            self.embedded_chunks = []
            return self.get_stats()

        # 2. Embed chunks
        self.embedded_chunks = embed_chunks(
            chunks=self.chunks,
            model_name=self.embedding_model,
        )

        # 3. Build FAISS Index
        self.vector_store.build(self.embedded_chunks)

        return self.get_stats()

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> GeneratedAnswer:
        """
        Executes end-to-end RAG question answering:
        Question -> Semantic Retrieval -> Grounded Context Prompt -> Gemini Answer.

        Args:
            question: Natural language question.
            top_k: Number of most relevant chunks to retrieve.

        Returns:
            GeneratedAnswer: Object containing question, answer text, and source references.

        Raises:
            RuntimeError: If called before a document is ingested and indexed.
            ValueError: If question is empty or invalid.
            LLMConfigurationError: If Gemini API key is missing.
            LLMGenerationError: If LLM call fails.
        """
        if not self.is_ready():
            raise RuntimeError(
                "No document has been processed or indexed yet. Ingest a PDF document first."
            )

        return generate_rag_answer(
            question=question,
            retrieval_service=self.retrieval_service,
            generator=self.generator,
            top_k=top_k,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Retrieves top_k chunks for a query without calling the LLM generator.

        Args:
            query: Query search string.
            top_k: Number of chunks to retrieve.

        Returns:
            List[SearchResult]: Retrieved search hits with similarity scores.
        """
        return self.retrieval_service.retrieve(query=query, top_k=top_k)

    def clear(self) -> None:
        """Resets document state and clears the FAISS vector index."""
        self.vector_store.clear()
        self.document = None
        self.chunks = []
        self.embedded_chunks = []
