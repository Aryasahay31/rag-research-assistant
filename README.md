# RAG Document Research Assistant

A production-ready Retrieval-Augmented Generation system that lets you upload documents and ask questions about them. Built with a hybrid search pipeline, cross-encoder reranking, and local LLM inference — no paid APIs required.

## Architecture

```
Document Upload → PDF/DOCX/TXT Loader → Text Chunking (800 tokens, 100 overlap)
                                              ↓
                                     Sentence-Transformers (all-MiniLM-L6-v2)
                                              ↓
                                        Qdrant Vector DB
                                              ↓
Query → [Optional HyDE] → Hybrid Search (Semantic + BM25 with RRF) → Cross-Encoder Reranking
                                                                            ↓
                                                                   Ollama LLM (Llama 3.2)
                                                                            ↓
                                                                    Answer + Sources
```

## Key Features

- **Hybrid Search** — Combines dense vector search (Qdrant) with sparse keyword search (BM25) using Reciprocal Rank Fusion (k=60)
- **Cross-Encoder Reranking** — ms-marco-MiniLM-L-6-v2 rescores candidates for higher precision
- **HyDE Query Expansion** — Generates hypothetical documents to improve retrieval for vague queries
- **Redis Caching** — TTL-based response caching with automatic invalidation on new document uploads
- **Conversation Memory** — Multi-turn context within a session
- **Evaluation Metrics** — Per-query latency breakdown, groundedness scoring, and source chunk inspection
- **Fully Local** — Ollama for generation, sentence-transformers for embeddings, no API keys needed

## Tech Stack

| Component | Technology |
|---|---|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, 384d) |
| Vector DB | Qdrant |
| Sparse Search | rank-bm25 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Ollama (Llama 3.2) |
| Cache | Redis |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker Compose, Kubernetes |

## Project Structure

```
rag-research-assistant/
├── app/
│   ├── config.py                # Settings via pydantic-settings
│   ├── main.py                  # FastAPI app entrypoint
│   ├── ingestion/
│   │   ├── loader.py            # PDF, DOCX, TXT document loading
│   │   ├── chunker.py           # Recursive text chunking
│   │   └── embedder.py          # Sentence-transformers + Qdrant storage
│   ├── retrieval/
│   │   ├── semantic_search.py   # Dense vector search via Qdrant
│   │   ├── bm25_search.py       # Sparse keyword search
│   │   ├── hybrid.py            # RRF fusion of semantic + BM25
│   │   ├── reranker.py          # Cross-encoder reranking
│   │   └── query_transform.py   # HyDE and multi-query expansion
│   ├── generation/
│   │   ├── llm.py               # Ollama chat completion
│   │   ├── prompt.py            # Prompt template builder
│   │   └── memory.py            # Conversation history
│   ├── cache/
│   │   └── redis_cache.py       # Redis caching layer
│   └── api/
│       └── routes.py            # REST endpoints with latency tracking
├── ui/
│   └── streamlit_app.py         # Chat UI with eval metrics
├── eval/
│   ├── evaluate.py              # RAGAS evaluation framework
│   ├── test_questions.json      # Evaluation question set
│   └── latency_report.json      # Pipeline latency benchmarks
├── eval_latency.py              # Per-stage latency profiler
├── k8s/
│   ├── configmap.yml
│   ├── deployment.yml
│   └── service.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

## Setup

### Prerequisites

- Python 3.11
- Docker Desktop (for Qdrant and Redis)
- Ollama (https://ollama.com)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Aryasahay31/rag-research-assistant.git
cd rag-research-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Qdrant and Redis:
```bash
docker-compose up -d qdrant redis
```

5. Pull the Ollama model:
```bash
ollama pull llama3.2
```

6. Start the backend:
```bash
uvicorn app.main:app --reload --port 8000
```

7. Start the frontend (new terminal):
```bash
streamlit run ui/streamlit_app.py
```

8. Open http://localhost:8501 in your browser.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload and index a document |
| POST | `/api/query` | Ask a question (returns answer, sources, latency, groundedness) |
| DELETE | `/api/documents` | Clear all indexed documents |
| GET | `/api/health` | Health check |

## Evaluation

Run the latency profiler:
```bash
python eval_latency.py
```

This measures per-stage latency (embedding, retrieval, reranking, generation) across multiple queries and saves results to `eval/latency_report.json`.

## Design Decisions

- **Hybrid search over pure semantic** — BM25 catches exact keyword matches that embeddings miss, and RRF fusion combines both rankings without tuning weights
- **Cross-encoder reranking** — Bi-encoders are fast but approximate; a cross-encoder jointly attends to query and document for better relevance scoring at the cost of higher latency on fewer candidates
- **HyDE as optional** — Useful for vague queries but adds an LLM call; disabled by default since direct queries already match well
- **Redis caching** — Identical queries skip the entire pipeline; cache invalidates automatically when new documents are indexed
- **Local inference** — No API costs or rate limits; production deployment would swap to GPU inference or a hosted endpoint

## License

MIT