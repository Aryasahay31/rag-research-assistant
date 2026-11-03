from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Ollama (replaces OpenAI)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2"

    # Qdrant vector database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "rag_documents"

    # Redis cache
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Chunking parameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100

    # Embedding (local sentence-transformers)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSIONS: int = 384  # MiniLM outputs 384-dim vectors

    # Retrieval
    SEMANTIC_TOP_K: int = 20
    BM25_TOP_K: int = 20
    RERANK_TOP_K: int = 5
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Cache TTL in seconds (1 hour)
    CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"


settings = Settings()