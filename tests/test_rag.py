"""
Unit and integration tests for RAG pipeline orchestration (Phase 7).
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.services.pdf_processor import ExtractedDocument, PageData
from backend.services.rag_pipeline import RAGPipeline
from backend.services.generator import GeneratedAnswer, LLMConfigurationError


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_extracted_document():
    """Create a sample ExtractedDocument with two pages of text."""
    return ExtractedDocument(
        file_path="test_guide.pdf",
        total_pages=2,
        total_characters=180,
        has_text=True,
        pages=[
            PageData(
                page_number=1,
                text="VeriDoc is an AI-powered document intelligence platform for PDF analysis.",
                char_count=75,
            ),
            PageData(
                page_number=2,
                text="The system uses FAISS vector indexing and sentence embeddings for semantic retrieval.",
                char_count=87,
            ),
        ],
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client returning a predictable response."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "VeriDoc uses FAISS vector indexing and sentence embeddings for retrieval."
    client.models.generate_content.return_value = mock_response
    return client


# -----------------------------------------------------------------------------
# Unit Tests for RAGPipeline
# -----------------------------------------------------------------------------

def test_rag_pipeline_initialization():
    """Test initial unpopulated state of RAGPipeline."""
    pipeline = RAGPipeline(api_key="fake-key-for-test")
    assert pipeline.is_ready() is False
    assert pipeline.document is None
    assert len(pipeline.chunks) == 0
    assert len(pipeline.vector_store) == 0

    stats = pipeline.get_stats()
    assert stats["is_ready"] is False
    assert stats["total_chunks"] == 0
    assert stats["indexed_vectors"] == 0


def test_rag_pipeline_ingest_document(sample_extracted_document, mock_llm_client):
    """Test end-to-end ingestion: chunking, embedding, and vector indexing."""
    pipeline = RAGPipeline(api_key="fake-key", llm_client=mock_llm_client)
    stats = pipeline.ingest_document(sample_extracted_document, chunk_size=500, chunk_overlap=100)

    assert stats["is_ready"] is True
    assert stats["document_name"] == "test_guide.pdf"
    assert stats["total_pages"] == 2
    assert stats["total_chunks"] >= 2
    assert stats["indexed_vectors"] >= 2
    assert pipeline.is_ready() is True


def test_rag_pipeline_ask_success(sample_extracted_document, mock_llm_client):
    """Test question answering flow through the ingested RAGPipeline."""
    pipeline = RAGPipeline(api_key="fake-key", llm_client=mock_llm_client)
    pipeline.ingest_document(sample_extracted_document, chunk_size=500, chunk_overlap=100)

    answer = pipeline.ask(
        question="What indexing technique does VeriDoc use?",
        top_k=2,
    )

    assert isinstance(answer, GeneratedAnswer)
    assert answer.question == "What indexing technique does VeriDoc use?"
    assert "FAISS" in answer.answer
    assert len(answer.sources) <= 2
    assert answer.sources[0].chunk.document_name == "test_guide.pdf"
    assert answer.sources[0].chunk.page_number in [1, 2]


def test_rag_pipeline_ask_before_ingestion_raises_error(mock_llm_client):
    """Test that calling ask() on an unpopulated pipeline raises RuntimeError."""
    pipeline = RAGPipeline(api_key="fake-key", llm_client=mock_llm_client)
    with pytest.raises(RuntimeError, match="No document has been processed or indexed yet"):
        pipeline.ask("What is this document about?")


def test_rag_pipeline_ask_with_empty_question_raises_error(sample_extracted_document, mock_llm_client):
    """Test that asking an empty question raises ValueError."""
    pipeline = RAGPipeline(api_key="fake-key", llm_client=mock_llm_client)
    pipeline.ingest_document(sample_extracted_document)

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        pipeline.ask("   ")


def test_rag_pipeline_retrieve_without_generation(sample_extracted_document, mock_llm_client):
    """Test direct chunk retrieval without invoking the LLM generator."""
    pipeline = RAGPipeline(api_key="fake-key", llm_client=mock_llm_client)
    pipeline.ingest_document(sample_extracted_document)

    hits = pipeline.retrieve("semantic retrieval embeddings", top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk.page_number == 2
    assert "embeddings" in hits[0].chunk.text
    # Ensure generator was not called
    mock_llm_client.models.generate_content.assert_not_called()


def test_rag_pipeline_clear(sample_extracted_document, mock_llm_client):
    """Test clearing and resetting the pipeline state."""
    pipeline = RAGPipeline(api_key="fake-key", llm_client=mock_llm_client)
    pipeline.ingest_document(sample_extracted_document)
    assert pipeline.is_ready() is True

    pipeline.clear()
    assert pipeline.is_ready() is False
    assert pipeline.document is None
    assert len(pipeline.chunks) == 0
    assert len(pipeline.vector_store) == 0


def test_rag_pipeline_ingest_empty_document(mock_llm_client):
    """Test that ingesting a document with no text handles safely."""
    empty_doc = ExtractedDocument(
        file_path="empty.pdf",
        total_pages=1,
        total_characters=0,
        has_text=False,
        pages=[PageData(page_number=1, text="", char_count=0)],
    )
    pipeline = RAGPipeline(api_key="fake-key", llm_client=mock_llm_client)
    stats = pipeline.ingest_document(empty_doc)

    assert stats["is_ready"] is False
    assert stats["total_chunks"] == 0
    assert stats["indexed_vectors"] == 0
    assert pipeline.is_ready() is False
