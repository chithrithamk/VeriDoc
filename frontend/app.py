"""
VeriDoc — Streamlit Frontend Application (Phase 2B)

Interactive user interface for uploading PDFs, extracting structured text,
chunking text on a per-page basis, and visually inspecting document chunks.
"""

from pathlib import Path
import sys
import uuid
import streamlit as st

# Ensure project root is in sys.path for backend module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriDoc — AI Document Intelligence Platform",
    page_icon="📄",
    layout="wide",
)

st.title("📄 VeriDoc — AI Document Intelligence Platform")
st.caption("Upload PDFs, extract page-level text, and inspect intelligent text chunks.")

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "extracted_doc" not in st.session_state:
    st.session_state.extracted_doc = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None
if "used_chunk_size" not in st.session_state:
    st.session_state.used_chunk_size = 1000
if "used_chunk_overlap" not in st.session_state:
    st.session_state.used_chunk_overlap = 200

# -----------------------------------------------------------------------------
# Sidebar: Document Upload & Chunking Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Select a PDF document (.pdf) to extract text and generate chunks.",
    )

    st.subheader("⚙️ Chunking Settings")
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

    if uploaded_file is not None:
        # Reset state if a new file is uploaded
        if st.session_state.processed_filename != uploaded_file.name:
            st.session_state.extracted_doc = None
            st.session_state.document_chunks = None
            st.session_state.processed_filename = None

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
                # Save uploaded file temporarily to disk
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Processing document and generating chunks..."):
                    # 1. PDF Text Extraction
                    extracted_doc = extract_text_from_pdf(temp_file_path)

                    # 2. Text Chunking
                    chunks = chunk_document(
                        extracted_doc,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )

                    # Store results in session state
                    st.session_state.extracted_doc = extracted_doc
                    st.session_state.document_chunks = chunks
                    st.session_state.processed_filename = uploaded_file.name
                    st.session_state.used_chunk_size = int(chunk_size)
                    st.session_state.used_chunk_overlap = int(chunk_overlap)

                st.success("Document processed and chunked successfully.")

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
                # Clean up temporary file
                if temp_file_path.exists():
                    try:
                        temp_file_path.unlink()
                    except Exception:
                        pass
    else:
        # Reset session state when no file is uploaded
        st.session_state.extracted_doc = None
        st.session_state.document_chunks = None
        st.session_state.processed_filename = None

# -----------------------------------------------------------------------------
# Main Display Area
# -----------------------------------------------------------------------------
if st.session_state.extracted_doc is not None and st.session_state.document_chunks is not None:
    doc: ExtractedDocument = st.session_state.extracted_doc
    chunks: list[DocumentChunk] = st.session_state.document_chunks

    st.subheader("📊 Document Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="File Name", value=st.session_state.processed_filename or "Unknown")
    with col2:
        st.metric(label="Total Pages", value=doc.total_pages)
    with col3:
        st.metric(label="Total Characters", value=doc.total_characters)
    with col4:
        st.metric(label="Total Chunks", value=len(chunks))

    st.divider()

    if not doc.has_text or doc.total_characters == 0:
        st.warning("No extractable text was found in this PDF.")
    else:
        # Create tabs for inspecting Pages vs Chunks
        tab_chunks, tab_pages = st.tabs(["✂️ Document Chunks", "📑 Extracted Pages"])

        with tab_chunks:
            st.subheader("✂️ Document Chunks")

            # Configuration metadata cards
            cfg1, cfg2, cfg3 = st.columns(3)
            with cfg1:
                st.info(f"**Total Chunks:** {len(chunks)}")
            with cfg2:
                st.info(f"**Chunk Size:** {st.session_state.used_chunk_size} characters")
            with cfg3:
                st.info(f"**Overlap:** {st.session_state.used_chunk_overlap} characters")

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
                        label_visibility="collapsed",
                    )

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
                            label_visibility="collapsed",
                        )
                    else:
                        st.caption("No extractable text on this page.")

elif uploaded_file is None:
    st.info("Upload a PDF document from the sidebar and click 'Process Document' to view extracted text and chunks.")
else:
    st.info("PDF selected. Click 'Process Document' in the sidebar to extract text and generate chunks.")
