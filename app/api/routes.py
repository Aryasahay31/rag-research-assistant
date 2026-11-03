"""
API Routes — the REST interface to our RAG pipeline.

Endpoints:
  POST /upload     — upload and ingest a document
  POST /query      — ask a question, get an answer + metrics
  GET  /health     — health check for Docker/K8s
  DELETE /documents — clear all indexed documents
  GET  /documents  — list indexed document sources
"""

import os
import time
import shutil
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.ingestion import DocumentLoader, TextChunker, DocumentEmbedder
from app.retrieval import HybridRetriever, CrossEncoderReranker, QueryTransformer
from app.generation import RAGChain
from app.cache import RedisCache


router = APIRouter()

# Initialize components (created once, reused across requests)
loader = DocumentLoader()
chunker = TextChunker()
embedder = DocumentEmbedder()
retriever = HybridRetriever()
reranker = CrossEncoderReranker()
query_transformer = QueryTransformer()
rag_chain = RAGChain()
cache = RedisCache()


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"
    use_hyde: bool = False


class ChunkDetail(BaseModel):
    text: str
    source: str
    page: str
    rerank_score: float


class LatencyMetrics(BaseModel):
    cache_check_ms: float = 0.0
    hyde_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list
    cached: bool = False
    latency: LatencyMetrics = LatencyMetrics()
    chunks_used: List[ChunkDetail] = []
    groundedness_score: float = 0.0


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str


# --- Helpers ---

def compute_groundedness(answer: str, chunks: List[Dict[str, Any]]) -> float:
    """
    Simple groundedness score: what fraction of answer sentences
    have keyword overlap with at least one source chunk.
    Score 0.0-1.0 (higher = more grounded in sources).
    """
    if not chunks or not answer.strip():
        return 0.0

    # Split answer into sentences
    sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    # Combine all chunk texts
    chunk_texts = " ".join(
        c.get("text", c.get("content", "")).lower() for c in chunks
    )

    # For each sentence, check if key words appear in chunks
    grounded_count = 0
    for sentence in sentences:
        words = set(sentence.lower().split())
        # Remove stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on",
                      "at", "to", "for", "of", "and", "or", "but", "not", "it",
                      "this", "that", "with", "from", "by", "as", "be", "has",
                      "have", "had", "do", "does", "did", "will", "can", "may",
                      "its", "also", "such", "which", "these", "those", "been",
                      "more", "than", "each", "about", "into", "over", "some"}
        keywords = words - stop_words
        if not keywords:
            grounded_count += 1
            continue

        # Count how many keywords appear in chunk text
        matches = sum(1 for w in keywords if w in chunk_texts)
        overlap = matches / len(keywords) if keywords else 0

        if overlap >= 0.3:  # at least 30% keyword overlap
            grounded_count += 1

    return round(grounded_count / len(sentences), 2)


# --- Endpoints ---

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, or TXT) for indexing."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in DocumentLoader.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {DocumentLoader.SUPPORTED_EXTENSIONS}",
        )

    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        documents = loader.load(file_path)
        chunks = chunker.chunk(documents)
        num_stored = embedder.embed_and_store(chunks)
        retriever.refresh_bm25()
        cache.clear_all()

        return UploadResponse(
            filename=file.filename,
            chunks_created=num_stored,
            message=f"Successfully indexed {file.filename} into {num_stored} chunks.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question — returns answer + latency + groundedness + source chunks."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    latency = LatencyMetrics()
    total_start = time.perf_counter()

    # 1. Cache check
    t0 = time.perf_counter()
    cached_result = cache.get(question)
    latency.cache_check_ms = round((time.perf_counter() - t0) * 1000, 1)

    if cached_result:
        latency.total_ms = round((time.perf_counter() - total_start) * 1000, 1)
        return QueryResponse(
            answer=cached_result["answer"],
            sources=cached_result["sources"],
            cached=True,
            latency=latency,
            chunks_used=cached_result.get("chunks_used", []),
            groundedness_score=cached_result.get("groundedness_score", 0.0),
        )

    try:
        # 2. HyDE (optional)
        search_query = question
        if request.use_hyde:
            t0 = time.perf_counter()
            search_query = query_transformer.hyde_transform(question)
            latency.hyde_ms = round((time.perf_counter() - t0) * 1000, 1)

        # 3. Hybrid retrieval
        t0 = time.perf_counter()
        candidates = retriever.search(search_query)
        latency.retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        # 4. Reranking
        t0 = time.perf_counter()
        top_chunks = reranker.rerank(question, candidates)
        latency.reranking_ms = round((time.perf_counter() - t0) * 1000, 1)

        # 5. Generation
        t0 = time.perf_counter()
        answer = rag_chain.generate(
            query=question,
            chunks=top_chunks,
            session_id=request.session_id,
        )
        latency.generation_ms = round((time.perf_counter() - t0) * 1000, 1)

        latency.total_ms = round((time.perf_counter() - total_start) * 1000, 1)

        # Extract source info
        sources = list({
            chunk["metadata"].get("source", "Unknown")
            for chunk in top_chunks
        })

        # Build chunk details for response
        chunks_used = []
        for chunk in top_chunks[:5]:
            chunks_used.append(ChunkDetail(
                text=chunk.get("text", chunk.get("content", ""))[:300],  # truncate
                source=chunk.get("metadata", {}).get("source", "Unknown"),
                page=str(chunk.get("metadata", {}).get("page", "N/A")),
                rerank_score=round(chunk.get("rerank_score", chunk.get("score", 0.0)), 4),
            ))

        # Compute groundedness
        groundedness = compute_groundedness(answer, top_chunks)

        # Cache the result (including metrics)
        result = {
            "answer": answer,
            "sources": sources,
            "chunks_used": [c.dict() for c in chunks_used],
            "groundedness_score": groundedness,
        }
        cache.set(question, result)

        return QueryResponse(
            answer=answer,
            sources=sources,
            cached=False,
            latency=latency,
            chunks_used=chunks_used,
            groundedness_score=groundedness,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents():
    """Clear all indexed documents and cache."""
    try:
        embedder.delete_collection()
        retriever.refresh_bm25()
        cache.clear_all()
        rag_chain.memory.clear_all()
        return {"message": "All documents and cache cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint for Docker/K8s probes."""
    return {
        "status": "healthy",
        "redis_connected": cache.is_connected(),
    }