"""
FastAPI Application — the entry point for our RAG API server.

Run with: uvicorn app.main:app --reload --port 8000

The --reload flag watches for file changes and auto-restarts
(only use in development, not production).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
    title="RAG Document Research Assistant",
    description="Upload documents and ask questions — powered by hybrid search + reranking",
    version="1.0.0",
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "app": "RAG Document Research Assistant",
        "docs": "/docs",  # FastAPI auto-generates Swagger UI
        "health": "/api/health",
    }