"""
Semantic Search — now using local sentence-transformers for query embedding.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from app.config import settings


class SemanticSearch:
    """Vector-based semantic search using Qdrant."""

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

    def _embed_query(self, query: str) -> List[float]:
        """Convert query text to a vector using local model."""
        embedding = self.model.encode(query)
        return embedding.tolist()

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        top_k = top_k or settings.SEMANTIC_TOP_K

        query_vector = self._embed_query(query)

        results = self.qdrant_client.search(
            collection_name=settings.COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
        )

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {
                    k: v for k, v in hit.payload.items() if k != "text"
                },
            }
            for hit in results
        ]