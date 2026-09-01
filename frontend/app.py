"""
VeriDoc — Streamlit Frontend Application (Phase 7)

Interactive user interface for:
1. Uploading PDF documents
2. Extracting structured page-level text (PyMuPDF)
3. Sentence-boundary-aware text chunking
4. Embedding generation & FAISS vector store indexing
5. Natural language Question Answering via Semantic Retrieval & Google Gemini
6. Transparent source citations and visual chunk inspection
"""

import os
from pathlib import Path
import sys
import uuid
import streamlit as st

# Ensure project root is in sys.path for backend module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_env_file() -> None:
    """Safely loads key-value pairs from .env if present into os.environ."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_env_file()

from backend.services.pdf_processor import (
    CorruptedPDFError,
    ExtractedDocument,
    InvalidPDFError,
    PDFNotFoundError,
    PDFProcessingError,
    extract_text_from_pdf,
)
from backend.services.chunker import (
    DocumentChunk,
    chunk_document,
)
from backend.services.embeddings import embed_chunks
from backend.services.vector_store import FAISSVectorStore, SearchResult
from backend.services.retrieval import RetrievalService
from backend.services.generator import (
    AnswerGenerator,
    GeneratedAnswer,
    LLMConfigurationError,
    LLMGenerationError,
    generate_rag_answer,
)

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriDoc — AI Document Intelligence Platform",
    page_icon="📄",
    layout="wide",
)

st.title("📄 VeriDoc — AI Document Intelligence Platform")
st.caption("Upload PDFs, extract and index text with FAISS, and ask natural language questions with Gemini.")

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "extracted_doc" not in st.session_state:
    st.session_state.extracted_doc = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = None
if "embedded_chunks" not in st.session_state:
    st.session_state.embedded_chunks = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None
if "used_chunk_size" not in st.session_state:
    st.session_state.used_chunk_size = 1000
if "used_chunk_overlap" not in st.session_state:
    st.session_state.used_chunk_overlap = 200
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []
if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = None

# -----------------------------------------------------------------------------
# Sidebar: Document Upload & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Select a PDF document (.pdf) to extract text, create embeddings, and build vector index.",
    )

    st.subheader("⚙️ Chunking & Search Settings")
    chunk_size = st.number_input(
        "Chunk Size (characters)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Target maximum character count per text chunk.",
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=1000,
        value=200,
        step=50,
        help="Number of overlapping characters between consecutive chunks on the same page.",
    )
    top_k = st.slider(
        "Top-K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of most semantically relevant chunks to retrieve for question answering.",
    )

    # API Key status indicator
    gemini_key_present = bool(os.getenv("GEMINI_API_KEY", "").strip())
    if gemini_key_present:
        st.success("🟢 Gemini API Key Detected")
    else:
        st.warning("⚠️ GEMINI_API_KEY missing from environment/.env")

    if uploaded_file is not None:
        # Reset state if a new file is uploaded
        if st.session_state.processed_filename != uploaded_file.name:
            st.session_state.extracted_doc = None
            st.session_state.document_chunks = None
            st.session_state.embedded_chunks = None
            st.session_state.vector_store = None
            st.session_state.processed_filename = None
            st.session_state.latest_answer = None
            st.session_state.qa_history = []

        process_button = st.button(
            "Process Document",
            type="primary",
            use_container_width=True,
        )

        if process_button:
            temp_dir = PROJECT_ROOT / "data" / "documents"
            temp_dir.mkdir(parents=True, exist_ok=True)
            safe_name = f"temp_{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
            temp_file_path = temp_dir / safe_name

            try:
                # Save uploaded file temporarily
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("1/3 Extracting page-level text..."):
                    extracted_doc = extract_text_from_pdf(temp_file_path)

                with st.spinner("2/3 Creating boundary-aligned chunks..."):
                    chunks = chunk_document(
                        extracted_doc,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )

                with st.spinner("3/3 Generating embeddings & building FAISS index..."):
                    if chunks:
                        embedded_chunks = embed_chunks(chunks)
                        store = FAISSVectorStore()
                        store.build(embedded_chunks)
                    else:
                        embedded_chunks = []
                        store = FAISSVectorStore()
                        store.build([])

                # Persist in session state
                st.session_state.extracted_doc = extracted_doc
                st.session_state.document_chunks = chunks
                st.session_state.embedded_chunks = embedded_chunks
                st.session_state.vector_store = store
                st.session_state.processed_filename = uploaded_file.name
                st.session_state.used_chunk_size = int(chunk_size)
                st.session_state.used_chunk_overlap = int(chunk_overlap)
                st.session_state.latest_answer = None

                st.success("Document extracted, chunked, and indexed in FAISS successfully!")

            except ValueError as err:
                st.error(f"Configuration Error: {err}")
            except PDFNotFoundError as err:
                st.error(f"File not found: {err}")
            except InvalidPDFError as err:
                st.error(f"Invalid PDF file: {err}")
            except CorruptedPDFError as err:
                st.error(f"Corrupted PDF file: {err}")
            except PDFProcessingError as err:
                st.error(f"PDF Processing Error: {err}")
            except Exception as err:
                st.error(f"An unexpected error occurred: {err}")
            finally:
                if temp_file_path.exists():
                    try:
                        temp_file_path.unlink()
                    except Exception:
                        pass
    else:
        # Reset session state when no file is uploaded
        st.session_state.extracted_doc = None
        st.session_state.document_chunks = None
        st.session_state.embedded_chunks = None
        st.session_state.vector_store = None
        st.session_state.processed_filename = None
        st.session_state.latest_answer = None

# -----------------------------------------------------------------------------
# Main Display Area
# -----------------------------------------------------------------------------
if st.session_state.extracted_doc is not None and st.session_state.vector_store is not None:
    doc: ExtractedDocument = st.session_state.extracted_doc
    chunks: list[DocumentChunk] = st.session_state.document_chunks or []
    store: FAISSVectorStore = st.session_state.vector_store

    # Summary Metrics Row
    st.subheader("📊 Document Information & Index Status")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="File Name", value=st.session_state.processed_filename or "Unknown")
    with col2:
        st.metric(label="Total Pages", value=doc.total_pages)
    with col3:
        st.metric(label="Characters", value=doc.total_characters)
    with col4:
        st.metric(label="Chunks", value=len(chunks))
    with col5:
        st.metric(label="Indexed Vectors", value=len(store))

    st.divider()

    if not doc.has_text or doc.total_characters == 0 or len(chunks) == 0:
        st.warning("No extractable text was found in this PDF. Vector index cannot be queried.")
    else:
        # Main Navigation Tabs: Q&A | Chunks | Pages
        tab_qa, tab_chunks, tab_pages = st.tabs([
            "💬 Ask Document (RAG)",
            "✂️ Document Chunks",
            "📑 Extracted Pages",
        ])

        # ---------------------------------------------------------------------
        # Tab 1: Q&A (RAG Pipeline)
        # ---------------------------------------------------------------------
        with tab_qa:
            st.subheader("💬 Ask Questions About Your Document")
            st.caption("Ask natural-language questions. VeriDoc retrieves relevant chunks and generates a grounded response.")

            # Question Input Form
            with st.form("rag_question_form", clear_on_submit=False):
                user_question = st.text_input(
                    "Your Question:",
                    placeholder="e.g., What are the main findings or objectives described in this document?",
                    key="question_input_field",
                )
                submit_button = st.form_submit_button("🔍 Search & Generate Answer", type="primary")

            if submit_button:
                if not user_question.strip():
                    st.warning("Please enter a question before searching.")
                elif not store.is_built() or len(store) == 0:
                    st.error("Vector index is not built. Please process a valid PDF first.")
                else:
                    try:
                        with st.spinner("Retrieving relevant chunks and generating answer..."):
                            retrieval_service = RetrievalService(vector_store=store)
                            generator = AnswerGenerator()

                            answer_result: GeneratedAnswer = generate_rag_answer(
                                question=user_question.strip(),
                                retrieval_service=retrieval_service,
                                generator=generator,
                                top_k=int(top_k),
                            )

                            st.session_state.latest_answer = answer_result
                            st.session_state.qa_history.insert(0, answer_result)

                    except LLMConfigurationError as err:
                        st.error(f"Configuration Error: {err}")
                    except LLMGenerationError as err:
                        st.error(f"LLM Generation Error: {err}")
                    except Exception as err:
                        st.error(f"An unexpected error occurred during Q&A: {err}")

            # Display Latest Answer & Citations
            if st.session_state.latest_answer is not None:
                ans: GeneratedAnswer = st.session_state.latest_answer

                st.markdown("### 💡 Answer")
                st.info(ans.answer)

                st.markdown("### 📚 Source Citations & Retrieved Context")
                if ans.sources:
                    for i, src in enumerate(ans.sources, start=1):
                        c = src.chunk
                        with st.expander(
                            f"Source #{i}: Page {c.page_number} (Chunk #{c.chunk_id}) | Similarity Score: {src.score:.3f} | {c.document_name}",
                            expanded=(i == 1),
                        ):
                            st.markdown(f"**Document:** `{c.document_name}` &nbsp;|&nbsp; **Page:** `{c.page_number}` &nbsp;|&nbsp; **Chunk ID:** `#{c.chunk_id}` &nbsp;|&nbsp; **Cosine Score:** `{src.score:.4f}`")
                            st.text_area(
                                label=f"Chunk {c.chunk_id} Content",
                                value=c.text,
                                height=120,
                                disabled=True,
                                key=f"ans_source_chunk_{i}_{c.chunk_id}",
                                label_visibility="collapsed",
                            )
                else:
                    st.caption("No relevant chunks were retrieved for this query.")

        # ---------------------------------------------------------------------
        # Tab 2: Document Chunks Visual Inspector
        # ---------------------------------------------------------------------
        with tab_chunks:
            st.subheader("✂️ Document Chunks")
            cfg1, cfg2, cfg3 = st.columns(3)
            with cfg1:
                st.info(f"**Total Chunks:** {len(chunks)}")
            with cfg2:
                st.info(f"**Chunk Size:** {st.session_state.used_chunk_size} chars")
            with cfg3:
                st.info(f"**Overlap:** {st.session_state.used_chunk_overlap} chars")

            for chunk in chunks:
                with st.expander(
                    f"Chunk #{chunk.chunk_id} | Page {chunk.page_number} ({chunk.char_count} characters)",
                    expanded=(chunk.chunk_id == 1),
                ):
                    st.text_area(
                        label=f"Chunk {chunk.chunk_id} Text",
                        value=chunk.text,
                        height=140,
                        disabled=True,
                        key=f"tab_chunk_{chunk.chunk_id}",
                        label_visibility="collapsed",
                    )

        # ---------------------------------------------------------------------
        # Tab 3: Raw Extracted Pages
        # ---------------------------------------------------------------------
        with tab_pages:
            st.subheader("📑 Raw Extracted Text (Page-by-Page)")
            for page in doc.pages:
                with st.expander(
                    f"Page {page.page_number} ({page.char_count} characters)",
                    expanded=(page.page_number == 1),
                ):
                    if page.text:
                        st.text_area(
                            label=f"Page {page.page_number} Content",
                            value=page.text,
                            height=160,
                            disabled=True,
                            key=f"tab_page_{page.page_number}",
                            label_visibility="collapsed",
                        )
                    else:
                        st.caption("No extractable text on this page.")

elif uploaded_file is None:
    st.info("Upload a PDF document from the sidebar and click 'Process Document' to begin.")
else:
    st.info("PDF selected. Click 'Process Document' in the sidebar to extract, chunk, and index the document.")
