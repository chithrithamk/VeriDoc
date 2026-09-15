"""
VeriDoc — Hallucination & Factual Support Checking Service (Phase 11)

This module provides a factual verification layer that checks whether a generated
answer is directly supported by the retrieved document evidence (chunks), identifying
supported vs unsupported claims and classifying the overall verification status.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import os
import re
from typing import Any, Dict, List, Optional
from google import genai

from backend.services.generator import DEFAULT_GEMINI_MODEL, LLMConfigurationError
from backend.services.vector_store import SearchResult


class SupportStatus(str, Enum):
    """Enumeration of possible support verification statuses."""
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VERIFICATION_UNAVAILABLE = "verification_unavailable"


@dataclass
class SupportCheckResult:
    """
    Structured result of factual support and hallucination verification.
    """
    status: str
    confidence: float
    explanation: str
    supported_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert support check result into a dictionary."""
        return {
            "status": self.status,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "supported_claims": list(self.supported_claims),
            "unsupported_claims": list(self.unsupported_claims),
        }


class SupportChecker:
    """
    Verifies candidate answers against retrieved document passages using Google Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        client: Optional[Any] = None,
    ):
        """
        Initialize the SupportChecker.

        Args:
            api_key: Optional Gemini API key. If omitted, resolved from GEMINI_API_KEY.
            model_name: Target Gemini model.
            client: Optional pre-configured client (for testing/mocking).
        """
        self.model_name = model_name

        if client is not None:
            self.client = client
        else:
            resolved_key = api_key or os.getenv("GEMINI_API_KEY")
            if not resolved_key or not resolved_key.strip():
                raise LLMConfigurationError(
                    "Gemini API key is not configured for SupportChecker. Set GEMINI_API_KEY in environment."
                )
            self.client = genai.Client(api_key=resolved_key.strip())

    def format_verification_prompt(
        self,
        question: str,
        answer: str,
        sources: List[SearchResult],
    ) -> str:
        """
        Constructs a strict verification prompt requiring JSON output.
        """
        evidence_blocks: List[str] = []
        for src in sources:
            chunk = src.chunk
            header = f"[Page {chunk.page_number} | Chunk #{chunk.chunk_id} | {chunk.document_name} | Similarity Score: {src.score:.4f}]"
            evidence_blocks.append(f"{header}\n{chunk.text}")

        evidence_text = "\n\n".join(evidence_blocks)

        prompt = (
            "DOCUMENT EVIDENCE:\n"
            "----------------------------------------\n"
            f"{evidence_text}\n"
            "----------------------------------------\n\n"
            "USER QUESTION:\n"
            f"{question.strip()}\n\n"
            "CANDIDATE ANSWER TO VERIFY:\n"
            f"{answer.strip()}\n\n"
            "TASK:\n"
            "You are a strict, objective factual verification assistant. Determine whether the CANDIDATE ANSWER "
            "is factually supported by the DOCUMENT EVIDENCE above.\n\n"
            "RULES:\n"
            "1. Treat the DOCUMENT EVIDENCE as the ONLY source of truth. Do NOT use outside knowledge.\n"
            "2. Break down the candidate answer into factual claims.\n"
            "3. For each claim, check if it is directly stated or explicitly entailed by the evidence.\n"
            "4. Choose exactly ONE status:\n"
            '   - "supported": Every factual claim in the answer is directly supported by the evidence.\n'
            '   - "partially_supported": Some claims are supported, but others are not found in or are contradicted by the evidence.\n'
            '   - "unsupported": The core claims in the answer are NOT supported by the evidence.\n'
            '   - "insufficient_evidence": The evidence is irrelevant, missing, or inadequate to verify the claims.\n'
            "5. Assign a confidence score between 0.0 and 1.0 reflecting certainty.\n"
            "6. List supported claims in 'supported_claims' and unsupported claims in 'unsupported_claims'.\n"
            "7. Provide a concise explanation.\n\n"
            "OUTPUT FORMAT:\n"
            "Respond ONLY with a valid raw JSON object matching this structure:\n"
            "{\n"
            '  "status": "supported" | "partially_supported" | "unsupported" | "insufficient_evidence",\n'
            '  "confidence": 0.95,\n'
            '  "explanation": "Concise explanation...",\n'
            '  "supported_claims": ["claim 1"],\n'
            '  "unsupported_claims": ["claim 2"]\n'
            "}\n"
            "Do NOT include markdown formatting or text outside the JSON object."
        )
        return prompt

    def _parse_json_response(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """
        Safely extracts and parses JSON from raw LLM text, stripping code blocks if present.
        """
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()

        # Remove markdown code block fences if present
        if text.startswith("```"):
            # Strip opening ```json or ```
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            # Strip closing ```
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: search for first { and last }
            match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None
            return None

    def check_support(
        self,
        question: str,
        answer: str,
        sources: List[SearchResult],
    ) -> SupportCheckResult:
        """
        Performs support verification of the answer against retrieved sources.

        Args:
            question: Original user query.
            answer: Candidate answer to check.
            sources: List of SearchResult objects retrieved from vector store.

        Returns:
            SupportCheckResult: Structured verification verdict.
        """
        clean_question = question.strip() if isinstance(question, str) else ""
        clean_answer = answer.strip() if isinstance(answer, str) else ""

        # Case 1: No evidence was retrieved
        if not sources:
            return SupportCheckResult(
                status=SupportStatus.INSUFFICIENT_EVIDENCE.value,
                confidence=1.0,
                explanation="No relevant document evidence was retrieved to support or verify this answer.",
                supported_claims=[],
                unsupported_claims=[clean_answer] if clean_answer else [],
            )

        # Case 2: Answer is empty or standard refusal
        if not clean_answer or "cannot be determined from the provided document context" in clean_answer.lower():
            return SupportCheckResult(
                status=SupportStatus.INSUFFICIENT_EVIDENCE.value,
                confidence=1.0,
                explanation="The model indicated that the document context was insufficient to answer the question.",
                supported_claims=[],
                unsupported_claims=[],
            )

        prompt = self.format_verification_prompt(clean_question, clean_answer, sources)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            raw_text = response.text.strip() if (response and getattr(response, "text", None)) else ""
            parsed = self._parse_json_response(raw_text)

            if not parsed or not isinstance(parsed, dict):
                return SupportCheckResult(
                    status=SupportStatus.VERIFICATION_UNAVAILABLE.value,
                    confidence=0.0,
                    explanation="Verification response could not be parsed as structured JSON.",
                    supported_claims=[],
                    unsupported_claims=[],
                )

            # Validate and extract status
            raw_status = str(parsed.get("status", "")).strip().lower()
            valid_statuses = {s.value for s in SupportStatus}
            status = raw_status if raw_status in valid_statuses else SupportStatus.VERIFICATION_UNAVAILABLE.value

            # Extract confidence (clamped 0.0 - 1.0)
            raw_conf = parsed.get("confidence", 0.0)
            try:
                confidence = max(0.0, min(1.0, float(raw_conf)))
            except (ValueError, TypeError):
                confidence = 0.5

            explanation = str(parsed.get("explanation", "Verification completed.")).strip()
            supported_claims = [str(c) for c in parsed.get("supported_claims", []) if str(c).strip()]
            unsupported_claims = [str(c) for c in parsed.get("unsupported_claims", []) if str(c).strip()]

            return SupportCheckResult(
                status=status,
                confidence=confidence,
                explanation=explanation,
                supported_claims=supported_claims,
                unsupported_claims=unsupported_claims,
            )

        except Exception as exc:
            # Gracefully handle any LLM/API errors without raising or breaking the pipeline
            return SupportCheckResult(
                status=SupportStatus.VERIFICATION_UNAVAILABLE.value,
                confidence=0.0,
                explanation=f"Support verification unavailable due to service error: {exc}",
                supported_claims=[],
                unsupported_claims=[],
            )
