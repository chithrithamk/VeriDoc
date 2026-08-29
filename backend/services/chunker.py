"""
VeriDoc — Natural Language Text Chunking Service (Phase 2)

This module splits extracted PDF document text into smaller chunks while preserving:
1. Natural sentence boundaries (. ? !)
2. Word boundaries (never splitting a word mid-character)
3. Sliding window overlap at natural word boundaries
4. Complete page-level and document-level metadata for citations
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from backend.services.pdf_processor import ExtractedDocument, PageData


@dataclass
class DocumentChunk:
    """
    Represents a discrete text chunk with associated document and page metadata.
    """
    chunk_id: int
    text: str
    page_number: int
    document_name: str
    char_count: int = 0

    def __post_init__(self):
        if self.char_count == 0 and self.text:
            self.char_count = len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_number": self.page_number,
            "document_name": self.document_name,
            "char_count": self.char_count,
        }


# -----------------------------------------------------------------------------
# Boundary Detection Helpers
# -----------------------------------------------------------------------------

def _find_sentence_end(text: str, start: int, target_end: int, min_end: int) -> int:
    """
    Find the last sentence-ending punctuation (. ? !) within text[start:target_end],
    ensuring it is at or after min_end and followed by whitespace or end-of-string.
    Returns the character index immediately following the punctuation, or -1 if none found.
    """
    sub = text[start:target_end]
    # Match sentence endings followed by optional quote and whitespace or end of string
    pattern = re.compile(r'([.!?]["\']?)(?:\s+|$)')
    matches = list(pattern.finditer(sub))

    for match in reversed(matches):
        end_idx = start + match.end(1)
        if end_idx >= min_end:
            return end_idx
    return -1


def _find_word_end(text: str, start: int, target_end: int, min_end: int) -> int:
    """
    Find the last whitespace character before target_end (at or after min_end).
    Returns the index where the word ends, or -1 if none found.
    """
    sub = text[start:target_end]
    for i in range(len(sub) - 1, -1, -1):
        if sub[i].isspace():
            end_idx = start + i
            if end_idx >= min_end:
                return end_idx
    return -1


def _find_word_start(text: str, target_pos: int, min_pos: int, max_pos: int) -> int:
    """
    Adjust a target overlap position to the nearest clean word boundary.
    """
    text_len = len(text)
    if target_pos <= min_pos:
        pos = min_pos
        while pos < max_pos and text[pos].isspace():
            pos += 1
        return pos

    # If target_pos is already at a word start (previous char is whitespace)
    if target_pos == 0 or text[target_pos - 1].isspace():
        pos = target_pos
        while pos < max_pos and text[pos].isspace():
            pos += 1
        return pos

    # Target is inside a word: search forward for next space, then word start
    pos = target_pos
    while pos < max_pos and not text[pos].isspace():
        pos += 1
    while pos < max_pos and text[pos].isspace():
        pos += 1
    if pos < max_pos:
        return pos

    # Fallback: search backward for word start if forward exceeded max_pos
    pos = target_pos
    while pos > min_pos and not text[pos - 1].isspace():
        pos -= 1
    while pos < max_pos and text[pos].isspace():
        pos += 1
    return pos


# -----------------------------------------------------------------------------
# Chunking Functions
# -----------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Splits text into chunks of at most `chunk_size` characters with `chunk_overlap` overlap,
    strictly respecting sentence and word boundaries.

    Args:
        text: Input string to split.
        chunk_size: Maximum target character length of each chunk.
        chunk_overlap: Desired number of overlapping characters between consecutive chunks.

    Returns:
        List[str]: List of non-empty text chunks.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be greater than or equal to 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )

    clean_text = text.strip()
    if not clean_text:
        return []

    # If the text fits completely within chunk_size, return it as a single chunk
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks: List[str] = []
    text_length = len(clean_text)
    start_idx = 0

    while start_idx < text_length:
        target_end = start_idx + chunk_size

        if target_end >= text_length:
            last_chunk = clean_text[start_idx:].strip()
            if last_chunk:
                chunks.append(last_chunk)
            break

        # 1. Prefer sentence boundary (. ? !) in the upper portion of the chunk window
        min_sentence_pos = start_idx + max(chunk_overlap, int(chunk_size * 0.4))
        cut_pos = _find_sentence_end(clean_text, start_idx, target_end, min_sentence_pos)

        # 2. Fallback to word boundary (whitespace) if no suitable sentence boundary is found
        if cut_pos == -1:
            min_word_pos = start_idx + 1
            cut_pos = _find_word_end(clean_text, start_idx, target_end, min_word_pos)

        # 3. Emergency fallback if no whitespace exists (e.g. uninterrupted sequence > chunk_size)
        if cut_pos == -1:
            cut_pos = target_end

        chunk_str = clean_text[start_idx:cut_pos].strip()
        if chunk_str:
            chunks.append(chunk_str)

        # Calculate overlap starting position at a clean word boundary
        target_overlap_start = max(start_idx + 1, cut_pos - chunk_overlap)
        next_start = _find_word_start(clean_text, target_overlap_start, start_idx + 1, cut_pos)

        # Guarantee strict forward progress to avoid infinite loops
        if next_start <= start_idx:
            next_start = cut_pos

        # Skip any leading whitespace for the start of the next chunk
        while next_start < text_length and clean_text[next_start].isspace():
            next_start += 1

        start_idx = next_start

    return chunks


def chunk_document(
    document: ExtractedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[DocumentChunk]:
    """
    Breaks an ExtractedDocument into structured DocumentChunks on a per-page basis,
    preserving natural language sentence/word boundaries and page citations.

    Args:
        document: ExtractedDocument instance containing pages and text.
        chunk_size: Target maximum characters per chunk (default: 1000).
        chunk_overlap: Desired number of overlapping characters between chunks (default: 200).

    Returns:
        List[DocumentChunk]: Sequential list of chunks with metadata attached.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be greater than or equal to 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )

    if not document or not document.pages:
        return []

    doc_name = Path(document.file_path).name if document.file_path else "document.pdf"
    if not doc_name.strip():
        doc_name = "document.pdf"

    document_chunks: List[DocumentChunk] = []
    chunk_counter = 1

    for page in document.pages:
        page_text = page.text.strip() if page.text else ""
        if not page_text:
            continue

        raw_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for text_segment in raw_chunks:
            chunk = DocumentChunk(
                chunk_id=chunk_counter,
                text=text_segment,
                page_number=page.page_number,
                document_name=doc_name,
                char_count=len(text_segment),
            )
            document_chunks.append(chunk)
            chunk_counter += 1

    return document_chunks
