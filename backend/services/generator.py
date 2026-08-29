"""
VeriDoc — RAG Answer Generation Service (Phase 6)

This module formats retrieved document chunks into a grounded context prompt and
calls the Google Gemini LLM via the official `google-genai` SDK to generate
accurate, context-constrained natural language answers with source references.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Union
from google import genai

from backend.services.retrieval import RetrievalService
from backend.services.vector_store import SearchResult


# Default recommended Gemini model for fast, high-quality responses
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# -----------------------------------------------------------------------------
# Custom Exceptions
# -----------------------------------------------------------------------------

class LLMConfigurationError(Exception):
    """Raised when the Gemini API key or model configuration is missing or invalid."""
    pass


class LLMGenerationError(Exception):
    """Raised when the LLM API call fails or encounters an error during generation."""
    pass


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class GeneratedAnswer:
    """
    Represents a complete grounded answer produced by the LLM alongside its source chunks.
    """
    question: str
    answer: str
    sources: List[SearchResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the answer and source metadata into a dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
        }


# -----------------------------------------------------------------------------
# Answer Generator Service
# -----------------------------------------------------------------------------

class AnswerGenerator:
    """
    Generates context-grounded answers using the Google Gemini LLM API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        client: Optional[Any] = None,
    ):
        """
        Initialize the AnswerGenerator.

        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY environment variable.
            model_name: Name of the Gemini model to invoke.
            client: Optional pre-configured genai.Client instance (useful for testing and dependency injection).

        Raises:
            LLMConfigurationError: If the API key is not provided and not found in environment.
        """
        self.model_name = model_name

        if client is not None:
            self.client = client
        else:
            resolved_key = api_key or os.getenv("GEMINI_API_KEY")
            if not resolved_key or not resolved_key.strip():
                raise LLMConfigurationError(
                    "Gemini API key is not configured. Set GEMINI_API_KEY in the environment or pass api_key to AnswerGenerator."
                )
            self.client = genai.Client(api_key=resolved_key.strip())

    def format_prompt(self, question: str, sources: List[SearchResult]) -> str:
        """
        Constructs a structured, grounded prompt constraining the LLM strictly to the provided context.

        Args:
            question: User's natural language question.
            sources: List of retrieved SearchResult objects.

        Returns:
            str: Formatted prompt string with context and strict instructions.
        """
        context_blocks: List[str] = []
        for src in sources:
            chunk = src.chunk
            header = f"[Page {chunk.page_number} | Chunk #{chunk.chunk_id} | {chunk.document_name}]"
            context_blocks.append(f"{header}\n{chunk.text}")

        context_text = "\n\n".join(context_blocks)

        prompt = (
            "DOCUMENT CONTEXT:\n"
            "----------------------------------------\n"
            f"{context_text}\n"
            "----------------------------------------\n\n"
            "USER QUESTION:\n"
            f"{question.strip()}\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question using ONLY the provided document context above.\n"
            "2. Do NOT invent information, speculate, or rely on outside knowledge.\n"
            "3. If the provided context does not contain enough information to answer the question, "
            'explicitly state: "The answer cannot be determined from the provided document context."\n'
            "4. Provide a direct, concise, and clear answer."
        )
        return prompt

    def generate_answer(
        self,
        question: str,
        sources: List[SearchResult],
    ) -> GeneratedAnswer:
        """
        Generates a grounded answer for a question based on retrieved sources.

        Args:
            question: User query string.
            sources: List of SearchResult objects retrieved from the vector store.

        Returns:
            GeneratedAnswer: Object containing the question, answer text, and source references.

        Raises:
            TypeError: If question is not a string.
            ValueError: If question is empty or whitespace-only.
            LLMGenerationError: If the Gemini API call fails.
        """
        if not isinstance(question, str):
            raise TypeError(f"Question must be a string, got {type(question).__name__}")

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty or whitespace-only.")

        # If no sources were retrieved, return the standard refusal without calling the API
        if not sources:
            return GeneratedAnswer(
                question=clean_question,
                answer="The answer cannot be determined from the provided document context.",
                sources=[],
            )

        prompt = self.format_prompt(clean_question, sources)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            raw_text = response.text.strip() if (response and getattr(response, "text", None)) else ""
            if not raw_text:
                raw_text = "The answer cannot be determined from the provided document context."

            return GeneratedAnswer(
                question=clean_question,
                answer=raw_text,
                sources=sources,
            )
        except Exception as exc:
            raise LLMGenerationError(
                f"Failed to generate answer from Gemini API: {exc}"
            ) from exc


# -----------------------------------------------------------------------------
# Orchestration Helper
# -----------------------------------------------------------------------------

def generate_rag_answer(
    question: str,
    retrieval_service: RetrievalService,
    generator: AnswerGenerator,
    top_k: int = 5,
) -> GeneratedAnswer:
    """
    Coordinates semantic retrieval and answer generation:
    1. Retrieves top_k relevant SearchResult objects from the retrieval service.
    2. Sends the question and retrieved sources to the AnswerGenerator.
    3. Returns the structured GeneratedAnswer.

    Args:
        question: User query string.
        retrieval_service: Initialized RetrievalService instance.
        generator: Initialized AnswerGenerator instance.
        top_k: Number of chunks to retrieve (default: 5).

    Returns:
        GeneratedAnswer: Answer text bundled with citation sources.
    """
    sources = retrieval_service.retrieve(query=question, top_k=top_k)
    return generator.generate_answer(question=question, sources=sources)
