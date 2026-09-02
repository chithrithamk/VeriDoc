"""
Tests for RAG Answer Generator service (Phase 6).
"""

from unittest.mock import MagicMock, patch
import os
import numpy as np
import pytest

from backend.services.embeddings import EmbeddedChunk
from backend.services.generator import (
    DEFAULT_GEMINI_MODEL,
    AnswerGenerator,
    GeneratedAnswer,
    LLMConfigurationError,
    LLMGenerationError,
    generate_rag_answer,
)
from backend.services.retrieval import RetrievalService
from backend.services.vector_store import FAISSVectorStore, SearchResult


# -----------------------------------------------------------------------------
# Test Fixtures & Mock Helpers
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_search_results():
    """Create sample SearchResult objects with page metadata for testing."""
    return [
        SearchResult(
            chunk=EmbeddedChunk(
                chunk_id=1,
                text="VeriDoc utilizes PyMuPDF for high-speed page-level text extraction.",
                page_number=1,
                document_name="spec.pdf",
                embedding=[1.0, 0.0],
                char_count=67,
            ),
            score=0.92,
        ),
        SearchResult(
            chunk=EmbeddedChunk(
                chunk_id=4,
                text="The embedding service uses sentence-transformers to create 384D dense vectors.",
                page_number=2,
                document_name="spec.pdf",
                embedding=[0.0, 1.0],
                char_count=77,
            ),
            score=0.85,
        ),
    ]


@pytest.fixture
def mock_genai_client():
    """Create a mock Google Gemini client with a mock response."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "VeriDoc uses PyMuPDF for text extraction and sentence-transformers for embeddings."
    client.models.generate_content.return_value = mock_response
    return client


# -----------------------------------------------------------------------------
# Unit Tests for AnswerGenerator
# -----------------------------------------------------------------------------

def test_generator_initialization_with_mock_client(mock_genai_client):
    """Test that AnswerGenerator initializes cleanly with an injected client."""
    generator = AnswerGenerator(client=mock_genai_client, model_name="gemini-3.6-flash")
    assert generator.client is mock_genai_client
    assert generator.model_name == "gemini-3.6-flash"


def test_generator_missing_api_key_raises_configuration_error():
    """Test that missing GEMINI_API_KEY raises a clean LLMConfigurationError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(LLMConfigurationError, match="Gemini API key is not configured"):
            AnswerGenerator(api_key=None, client=None)


def test_generate_answer_success(mock_genai_client, sample_search_results):
    """Test standard answer generation with question and retrieved chunks."""
    generator = AnswerGenerator(client=mock_genai_client)

    result = generator.generate_answer(
        question="How does VeriDoc extract text?",
        sources=sample_search_results,
    )

    assert isinstance(result, GeneratedAnswer)
    assert result.question == "How does VeriDoc extract text?"
    assert "PyMuPDF" in result.answer
    assert len(result.sources) == 2
    assert result.sources[0].chunk.page_number == 1
    assert result.sources[1].chunk.page_number == 2

    # Verify the client called generate_content with correct model
    mock_genai_client.models.generate_content.assert_called_once()
    call_kwargs = mock_genai_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_GEMINI_MODEL
    assert "DOCUMENT CONTEXT" in call_kwargs["contents"]


def test_format_prompt_contains_context_and_page_metadata(mock_genai_client, sample_search_results):
    """Test that the prompt builder formats chunks with page numbers and instructions."""
    generator = AnswerGenerator(client=mock_genai_client)
    prompt = generator.format_prompt("What embeddings are used?", sample_search_results)

    # Verify prompt structure and constraints
    assert "DOCUMENT CONTEXT:" in prompt
    assert "[Page 1 | Chunk #1 | spec.pdf]" in prompt
    assert "[Page 2 | Chunk #4 | spec.pdf]" in prompt
    assert "PyMuPDF for high-speed page-level text extraction" in prompt
    assert "sentence-transformers to create 384D dense vectors" in prompt
    assert "USER QUESTION:\nWhat embeddings are used?" in prompt
    assert "INSTRUCTIONS:" in prompt
    assert "Answer the user's question using ONLY the provided document context above." in prompt


def test_empty_question_raises_value_error(mock_genai_client, sample_search_results):
    """Test that empty or whitespace-only questions raise ValueError."""
    generator = AnswerGenerator(client=mock_genai_client)

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        generator.generate_answer("", sample_search_results)

    with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
        generator.generate_answer("   \n\t  ", sample_search_results)


def test_non_string_question_raises_type_error(mock_genai_client, sample_search_results):
    """Test that passing a non-string question raises TypeError."""
    generator = AnswerGenerator(client=mock_genai_client)
    with pytest.raises(TypeError, match="Question must be a string"):
        generator.generate_answer(12345, sample_search_results)  # type: ignore


def test_empty_sources_returns_refusal_without_calling_llm(mock_genai_client):
    """Test that empty retrieval results return a refusal message without calling the API."""
    generator = AnswerGenerator(client=mock_genai_client)

    result = generator.generate_answer(
        question="What is the capital of Mars?",
        sources=[],
    )

    assert "The answer cannot be determined" in result.answer
    assert result.sources == []
    mock_genai_client.models.generate_content.assert_not_called()


def test_llm_api_failure_raises_llm_generation_error(sample_search_results):
    """Test that API exceptions are wrapped cleanly into LLMGenerationError."""
    failing_client = MagicMock()
    failing_client.models.generate_content.side_effect = RuntimeError("Rate limit or connection timeout")

    generator = AnswerGenerator(client=failing_client)

    with pytest.raises(LLMGenerationError, match="Failed to generate answer from Gemini API"):
        generator.generate_answer("Test question", sample_search_results)


def test_generated_answer_to_dict(sample_search_results):
    """Test dictionary serialization of GeneratedAnswer."""
    answer = GeneratedAnswer(
        question="What is VeriDoc?",
        answer="VeriDoc is an AI document intelligence platform.",
        sources=sample_search_results,
    )
    data = answer.to_dict()

    assert data["question"] == "What is VeriDoc?"
    assert data["answer"] == "VeriDoc is an AI document intelligence platform."
    assert len(data["sources"]) == 2
    assert data["sources"][0]["chunk"]["page_number"] == 1
    assert data["sources"][0]["score"] == 0.92


# -----------------------------------------------------------------------------
# End-to-End Orchestration Integration Test
# -----------------------------------------------------------------------------

def test_rag_pipeline_orchestration_integration(mock_genai_client):
    """
    Test the full integration flow:
    Question -> RetrievalService -> SearchResults -> AnswerGenerator -> Final Grounded Answer
    """
    # 1. Setup deterministic vector store
    chunks = [
        EmbeddedChunk(
            chunk_id=1,
            text="The project is called VeriDoc, an AI document platform.",
            page_number=1,
            document_name="doc.pdf",
            embedding=[1.0, 0.0, 0.0, 0.0],
            char_count=56,
        ),
        EmbeddedChunk(
            chunk_id=2,
            text="VeriDoc uses FAISS vector indexing for similarity retrieval.",
            page_number=2,
            document_name="doc.pdf",
            embedding=[0.0, 1.0, 0.0, 0.0],
            char_count=60,
        ),
    ]
    store = FAISSVectorStore()
    store.build(chunks)

    # 2. Setup RetrievalService
    retrieval_service = RetrievalService(vector_store=store)

    # 3. Setup AnswerGenerator with mocked client
    generator = AnswerGenerator(client=mock_genai_client)

    # Mock embed_text in retrieval so query vector matches Chunk #2
    with patch("backend.services.retrieval.embed_text", return_value=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)):
        answer_result = generate_rag_answer(
            question="What vector indexing does VeriDoc use?",
            retrieval_service=retrieval_service,
            generator=generator,
            top_k=1,
        )

        assert isinstance(answer_result, GeneratedAnswer)
        assert answer_result.question == "What vector indexing does VeriDoc use?"
        assert len(answer_result.sources) == 1
        assert answer_result.sources[0].chunk.chunk_id == 2
        assert answer_result.sources[0].chunk.page_number == 2
        assert "FAISS vector indexing" in answer_result.sources[0].chunk.text
