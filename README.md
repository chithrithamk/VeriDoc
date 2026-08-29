# VeriDoc — AI Document Intelligence Platform

VeriDoc is an AI-powered document intelligence platform designed to extract knowledge from PDF documents using Retrieval-Augmented Generation (RAG). Users can upload documents, query them in natural language, and receive grounded answers with exact source citations and support verification.

---

## 🚀 Key Features

- **Document Ingestion & Processing**: PDF parsing, text extraction, and metadata extraction via PyMuPDF.
- **Smart Chunking**: Configurable text splitting with sliding-window chunk overlap.
- **Vector Search Engine**: Dense embeddings and FAISS index for high-speed semantic retrieval.
- **Context-Grounded Q&A**: LLM-powered answering constrained strictly to retrieved context.
- **Source & Page Citations**: Transparent page-level references for all generated answers.
- **Hallucination Checking**: Basic support verification mechanism to ensure answer fidelity.
- **Interactive UI & APIs**: Streamlit web dashboard and FastAPI REST backend.

---

## 🏗️ Architecture & RAG Pipeline

```text
PDF Document
    │
    ▼
Text Extraction (PyMuPDF)
    │
    ▼
Chunking & Metadata Tagging
    │
    ▼
Vector Embeddings
    │
    ▼
FAISS Vector Store (Indexed Search)
    │
    ▼
Similarity Retrieval (Top-K Chunks)
    │
    ▼
LLM Generation + Hallucination Check
    │
    ▼
Grounded Answer + Source Citations
```

---

## 📂 Project Structure

```text
VeriDoc/
│
├── backend/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── api/                        # API route handlers
│   │   ├── __init__.py
│   │   ├── documents.py            # Document upload & management endpoints
│   │   └── questions.py            # Query & Q&A endpoints
│   │
│   ├── services/                   # Core business & RAG services
│   │   ├── __init__.py
│   │   ├── pdf_processor.py        # PDF text extraction
│   │   ├── chunker.py              # Text chunking logic
│   │   ├── embeddings.py           # Embedding generation
│   │   ├── vector_store.py         # FAISS vector store management
│   │   ├── rag_pipeline.py         # End-to-end RAG orchestrator
│   │   └── llm_service.py          # LLM interface & prompt handling
│   │
│   └── models/                     # Data schemas & validation
│       ├── __init__.py
│       └── schemas.py              # Pydantic request/response models
│
├── frontend/
│   └── app.py                      # Streamlit interactive UI
│
├── data/
│   ├── documents/                  # Uploaded PDF document storage
│   └── vector_store/               # FAISS indices & chunk metadata
│
├── tests/
│   ├── __init__.py
│   ├── test_pdf_processor.py       # PDF processing unit tests
│   ├── test_chunker.py             # Chunking unit tests
│   └── test_rag.py                 # RAG pipeline unit tests
│
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── Dockerfile                      # Container definition
```

---

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit
- **PDF Extraction**: PyMuPDF (`fitz`)
- **Vector Search**: FAISS (`faiss-cpu`), NumPy
- **Embeddings & LLM**: OpenAI API / Gemini API
- **Testing**: pytest, pytest-asyncio, httpx
- **Containerization**: Docker

---

## 🚦 Incremental Roadmap

- [ ] **Phase 1**: PDF Text Extraction
- [ ] **Phase 2**: Text Chunking
- [ ] **Phase 3**: Embeddings Generation
- [ ] **Phase 4**: FAISS Vector Store
- [ ] **Phase 5**: Semantic Retrieval
- [ ] **Phase 6**: LLM Integration
- [ ] **Phase 7**: RAG Pipeline Orchestration
- [ ] **Phase 8**: FastAPI Backend
- [ ] **Phase 9**: Streamlit Frontend
- [ ] **Phase 10**: Source & Page Citations
- [ ] **Phase 11**: Hallucination / Support Checking
- [ ] **Phase 12**: SQLite History Storage
- [ ] **Phase 13**: Automated Testing Suite
- [ ] **Phase 14**: Docker Containerization
- [ ] **Phase 15**: Documentation & Portfolio Polish
