# ==============================================================================
# VeriDoc — Production Dockerfile (Phase 14)
# ==============================================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install minimal OS dependencies for PyMuPDF/FAISS/curl health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SentenceTransformer embedding model and CrossEncoder into the image layer
# This ensures zero-latency cold starts and full offline execution
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('intfloat/e5-small-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy application source code
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY README.md pyproject.toml /app/

# Create data directories with appropriate permissions
RUN mkdir -p /app/data/documents /app/data/vector_store /app/.cache

# Create non-root user and assign permissions
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose ports: 8000 (FastAPI Backend) & 8501 (Streamlit Frontend)
EXPOSE 8000
EXPOSE 8501

# Default command: Start FastAPI REST API server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
