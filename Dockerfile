# ==============================================================================
# VeriDoc — Multi-Stage / Base Dockerfile
# ==============================================================================

FROM python:3.11-slim

# Set working directory and environment variables
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase
COPY . .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command: run FastAPI backend (can be overridden for Streamlit)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
