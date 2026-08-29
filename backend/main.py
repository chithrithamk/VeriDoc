"""
VeriDoc — FastAPI Backend Application Entrypoint
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI(
    title="VeriDoc API",
    description="AI Document Intelligence Platform — RAG Backend",
    version="0.1.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check / root endpoint."""
    return {"message": "VeriDoc API is running", "status": "healthy"}


# TODO: Register routers in Phase 8
# from backend.api.documents import router as documents_router
# from backend.api.questions import router as questions_router
# app.include_router(documents_router, prefix="/api/v1/documents", tags=["documents"])
# app.include_router(questions_router, prefix="/api/v1/questions", tags=["questions"])
