"""
VeriDoc — Streamlit Frontend Application (Phases 7 & 10)

Interactive user interface for:
1. Uploading PDF documents
2. Extracting structured page-level text (PyMuPDF)
3. Sentence-boundary-aware text chunking
4. Embedding generation & FAISS vector store indexing
5. Multi-question natural language Q&A via Semantic Retrieval & Google Gemini
6. Transparent source citations with page numbers, similarity scores, and session history
"""

from datetime import datetime
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
from backend.services.support_checker import (
    SupportChecker,
    SupportCheckResult,
    SupportStatus,
)
from backend.database import (
    clear_all_question_records,
    count_question_records,
    create_document_record,
    create_question_record,
    get_latest_document_record,
    init_db,
    list_question_records,
    list_questions_for_document,
)
from backend.database.session import get_db

# Ensure database tables exist
init_db()


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriDoc — AI Document Intelligence Platform",
    page_icon="📄",
    layout="wide",
)

st.title("📄 VeriDoc — AI Document Intelligence Platform")
st.caption("Upload PDFs, index text in FAISS, and ask multiple natural language questions with source citations.")

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
    st.session_state.used_chunk_size = 400
if "used_chunk_overlap" not in st.session_state:
    st.session_state.used_chunk_overlap = 80
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
        value=400,
        step=50,
        help="Target maximum character count per text chunk.",
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=1000,
        value=80,
        step=20,
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

                # Persist in session state and reset Q&A history for the new document
                st.session_state.extracted_doc = extracted_doc
                st.session_state.document_chunks = chunks
                st.session_state.embedded_chunks = embedded_chunks
                st.session_state.vector_store = store
                st.session_state.processed_filename = uploaded_file.name
                st.session_state.used_chunk_size = int(chunk_size)
                st.session_state.used_chunk_overlap = int(chunk_overlap)
                st.session_state.latest_answer = None
                st.session_state.qa_history = []

                # Persist document metadata in SQLite database
                try:
                    with next(get_db()) as db:
                        create_document_record(
                            db=db,
                            filename=uploaded_file.name,
                            total_pages=extracted_doc.total_pages,
                            total_characters=extracted_doc.total_characters,
                            total_chunks=len(chunks),
                            indexed_vectors=len(store),
                            file_size_bytes=getattr(uploaded_file, "size", 0) or 0,
                            status="indexed",
                        )
                except Exception:
                    pass

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
        st.session_state.qa_history = []

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
        # Main Navigation Tabs: Q&A | Persistent History | Chunks | Pages
        tab_qa, tab_history, tab_chunks, tab_pages = st.tabs([
            "💬 Ask Document (RAG)",
            "📜 Persistent Q&A History",
            "✂️ Document Chunks",
            "📑 Extracted Pages",
        ])


        # ---------------------------------------------------------------------
        # Tab 1: Q&A (RAG Pipeline with Source Citations & History)
        # ---------------------------------------------------------------------
        with tab_qa:
            st.subheader("💬 Ask Questions About Your Document")
            st.caption("Ask natural-language questions. You can ask multiple questions without re-processing the document.")

            # Question Input Form (Input remains editable and re-submittable)
            with st.form("rag_question_form", clear_on_submit=False):
                user_question = st.text_input(
                    "Your Question:",
                    placeholder="e.g., What are the main findings or objectives described in this document?",
                    key="question_input_field",
                    help="Type your question. You can edit, erase, and ask another question at any time.",
                )
                submit_button = st.form_submit_button("🔍 Search & Generate Answer", type="primary")

            if submit_button:
                if not user_question.strip():
                    st.warning("Please enter a question before searching.")
                elif not store.is_built() or len(store) == 0:
                    st.error("Vector index is not built. Please process a valid PDF first.")
                else:
                    try:
                        with st.spinner("Retrieving relevant chunks, generating answer & verifying support..."):
                            retrieval_service = RetrievalService(vector_store=store)
                            generator = AnswerGenerator()
                            support_checker = SupportChecker()

                            # 1. Retrieve evidence
                            sources = retrieval_service.retrieve(query=user_question.strip(), top_k=int(top_k))

                            # 2. Generate grounded answer
                            answer_result: GeneratedAnswer = generator.generate_answer(
                                question=user_question.strip(),
                                sources=sources,
                            )

                            # 3. Verify factual support against retrieved evidence
                            support_result: SupportCheckResult = support_checker.check_support(
                                question=user_question.strip(),
                                answer=answer_result.answer,
                                sources=sources,
                            )
                            answer_result.support = support_result

                            st.session_state.latest_answer = answer_result
                            st.session_state.qa_history.insert(0, {
                                "question": user_question.strip(),
                                "answer": answer_result.answer,
                                "sources": answer_result.sources,
                                "support": support_result,
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                            })

                            # 4. Persist Q&A record into SQLite database
                            try:
                                with next(get_db()) as db:
                                    latest_doc = get_latest_document_record(db)
                                    doc_id = latest_doc.id if latest_doc else None
                                    doc_name = latest_doc.filename if latest_doc else (st.session_state.processed_filename or "document.pdf")
                                    sources_dicts = [
                                        {
                                            "chunk_id": s.chunk.chunk_id,
                                            "page_number": s.chunk.page_number,
                                            "document_name": s.chunk.document_name,
                                            "char_count": s.chunk.char_count,
                                            "text": s.chunk.text,
                                            "similarity_score": float(s.score),
                                        }
                                        for s in sources
                                    ]
                                    create_question_record(
                                        db=db,
                                        question=user_question.strip(),
                                        answer=answer_result.answer,
                                        document_id=doc_id,
                                        document_name=doc_name,
                                        sources=sources_dicts,
                                        support_status=support_result.status if support_result else None,
                                        support_confidence=support_result.confidence if support_result else None,
                                        support_explanation=support_result.explanation if support_result else None,
                                        supported_claims=support_result.supported_claims if support_result else [],
                                        unsupported_claims=support_result.unsupported_claims if support_result else [],
                                    )
                            except Exception:
                                pass

                    except LLMConfigurationError as err:
                        st.error(f"Configuration Error: {err}")
                    except LLMGenerationError as err:
                        st.error(f"LLM Generation Error: {err}. Please check your quota or try again.")
                    except Exception as err:
                        st.error(f"An unexpected error occurred during Q&A: {err}")


            # Display Latest Answer & Verification Status
            if st.session_state.latest_answer is not None:
                ans: GeneratedAnswer = st.session_state.latest_answer

                st.markdown("### 💡 Grounded Answer")
                st.info(ans.answer)

                # Display Factual Support Verification Result
                if ans.support is not None:
                    supp: SupportCheckResult = ans.support
                    status = supp.status
                    conf_pct = int(supp.confidence * 100) if supp.confidence <= 1.0 else int(supp.confidence)

                    st.markdown("### 🛡️ Factual Support & Hallucination Check")
                    if status == "supported":
                        st.success(f"✅ **Answer Supported by Document** (Confidence: {conf_pct}%)")
                    elif status == "partially_supported":
                        st.warning(f"🟡 **Partially Supported** (Confidence: {conf_pct}%)")
                    elif status == "unsupported":
                        st.error(f"⚠️ **Answer Not Fully Supported** (Confidence: {conf_pct}%)")
                    elif status == "insufficient_evidence":
                        st.info(f"🔍 **Insufficient Evidence** (Confidence: {conf_pct}%)")
                    else:
                        st.info(f"ℹ️ **Verification Status:** `{status}`")

                    if supp.explanation:
                        st.caption(f"**Analysis:** {supp.explanation}")

                    if supp.unsupported_claims:
                        with st.expander("⚠️ Unsupported / Unverified Claims", expanded=True):
                            for claim in supp.unsupported_claims:
                                st.markdown(f"- ❌ {claim}")

                    if supp.supported_claims and status != "supported":
                        with st.expander("✅ Verified Supported Claims", expanded=False):
                            for claim in supp.supported_claims:
                                st.markdown(f"- ✔️ {claim}")

                st.markdown("### 📚 Sources / Evidence")
                if ans.sources:
                    for i, src in enumerate(ans.sources, start=1):
                        c = src.chunk
                        with st.expander(
                            f"Source {i} — Page {c.page_number} | Document: {c.document_name} (Similarity: {src.score:.3f})",
                            expanded=(i == 1),
                        ):
                            st.markdown(
                                f"**Document:** `{c.document_name}` &nbsp;|&nbsp; "
                                f"**Page:** `{c.page_number}` &nbsp;|&nbsp; "
                                f"**Chunk ID:** `#{c.chunk_id}` &nbsp;|&nbsp; "
                                f"**Similarity Score:** `{src.score:.4f}`"
                            )
                            st.text_area(
                                label=f"Source {i} Text",
                                value=c.text,
                                height=120,
                                disabled=True,
                                key=f"ans_source_chunk_{i}_{c.chunk_id}",
                                label_visibility="collapsed",
                            )
                else:
                    st.caption("No relevant chunks were retrieved for this query.")

            # Display Session Q&A History
            if len(st.session_state.qa_history) > 1:
                st.divider()
                h_col1, h_col2 = st.columns([4, 1])
                with h_col1:
                    st.markdown("### 📜 Q&A Session History")
                with h_col2:
                    if st.button("🗑️ Clear History", key="btn_clear_history"):
                        st.session_state.qa_history = []
                        st.session_state.latest_answer = None
                        st.rerun()

                for h_idx, item in enumerate(st.session_state.qa_history[1:], start=1):
                    with st.expander(f"Q: {item['question']} ({item.get('timestamp', '')})", expanded=False):
                        if item.get("support"):
                            s_obj = item["support"]
                            s_status = s_obj.status if hasattr(s_obj, "status") else s_obj.get("status", "")
                            s_icon = "✅" if s_status == "supported" else ("🟡" if s_status == "partially_supported" else ("⚠️" if s_status == "unsupported" else "🔍"))
                            st.markdown(f"**Verification Status:** {s_icon} `{s_status}`")

                        st.markdown(f"**Answer:** {item['answer']}")
                        if item.get("sources"):
                            st.markdown("**Evidence / Sources:**")
                            for s_idx, s in enumerate(item["sources"], start=1):
                                sc = s.chunk
                                st.markdown(
                                    f"- **Source {s_idx} (Page {sc.page_number}, Chunk #{sc.chunk_id}):** "
                                    f"{sc.text[:200]}... *(Score: {s.score:.3f})*"
                                )

        # ---------------------------------------------------------------------
        # Tab 2: Persistent Q&A History (Loaded directly from SQLite)
        # ---------------------------------------------------------------------
        with tab_history:
            st.subheader("📜 Persistent Q&A History (SQLite Database)")
            st.caption("Historical questions, answers, support checks, and citations persisted in SQLite. Loaded directly without re-querying Gemini or FAISS.")

            hist_col1, hist_col2 = st.columns([4, 1])
            with hist_col2:
                if st.button("🗑️ Clear Persistent History", key="btn_clear_persistent_history", type="secondary"):
                    try:
                        with next(get_db()) as db:
                            cleared_count = clear_all_question_records(db)
                        st.session_state.qa_history = []
                        st.session_state.latest_answer = None
                        st.success(f"Cleared {cleared_count} question records from SQLite database.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear history: {e}")

            try:
                with next(get_db()) as db:
                    db_questions = list_question_records(db, limit=50)
                    total_saved = count_question_records(db)
            except Exception:
                db_questions = []
                total_saved = 0

            with hist_col1:
                st.info(f"**Total Persisted Questions in Database:** {total_saved}")

            if not db_questions:
                st.info("No persistent Q&A records found in the database. Ask a question to start recording history.")
            else:
                for q_idx, q_rec in enumerate(db_questions, start=1):
                    created_str = q_rec.created_at.strftime("%Y-%m-%d %H:%M:%S") if q_rec.created_at else ""
                    status_icon = "✅" if q_rec.support_status == "supported" else (
                        "🟡" if q_rec.support_status == "partially_supported" else (
                            "⚠️" if q_rec.support_status == "unsupported" else "🔍"
                        )
                    )
                    with st.expander(
                        f"#{q_idx} | {status_icon} Q: {q_rec.question} ({q_rec.document_name or 'Doc'} • {created_str})",
                        expanded=(q_idx == 1),
                    ):
                        st.markdown(f"**Question:** {q_rec.question}")
                        st.markdown(f"**Answer:** {q_rec.answer}")
                        st.markdown(
                            f"**Document:** `{q_rec.document_name or 'N/A'}` &nbsp;|&nbsp; "
                            f"**Saved At:** `{created_str}` &nbsp;|&nbsp; "
                            f"**Record ID:** `{q_rec.id}`"
                        )

                        # Support verification details
                        if q_rec.support_status:
                            st.markdown("---")
                            conf_pct = int((q_rec.support_confidence or 0.0) * 100) if (q_rec.support_confidence or 0.0) <= 1.0 else int(q_rec.support_confidence or 0.0)
                            st.markdown(f"**Support Verification:** {status_icon} `{q_rec.support_status}` (Confidence: {conf_pct}%)")
                            if q_rec.support_explanation:
                                st.caption(f"**Explanation:** {q_rec.support_explanation}")
                            if q_rec.unsupported_claims:
                                st.markdown("**Unsupported Claims:**")
                                for claim in q_rec.unsupported_claims:
                                    st.markdown(f"- ❌ {claim}")
                            if q_rec.supported_claims and q_rec.support_status != "supported":
                                st.markdown("**Supported Claims:**")
                                for claim in q_rec.supported_claims:
                                    st.markdown(f"- ✔️ {claim}")

                        # Citations / Sources
                        if q_rec.sources_data:
                            st.markdown("---")
                            st.markdown("**Evidence / Sources:**")
                            for s_idx, src in enumerate(q_rec.sources_data, start=1):
                                p_num = src.get("page_number", "N/A")
                                c_id = src.get("chunk_id", "N/A")
                                score = src.get("similarity_score", 0.0)
                                text_preview = src.get("text", "")
                                st.markdown(
                                    f"- **Source {s_idx} (Page {p_num}, Chunk #{c_id}, Score: {score:.3f}):** "
                                    f"{text_preview[:250]}..."
                                )

        # ---------------------------------------------------------------------
        # Tab 3: Document Chunks Visual Inspector
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
        # Tab 4: Raw Extracted Pages
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
