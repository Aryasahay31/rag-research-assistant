"""
Document Embedder — now using local sentence-transformers.
No API key needed. The model runs on your machine (CPU is fine).
all-MiniLM-L6-v2 outputs 384-dimensional vectors.
"""

import uuid
from typing import List
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from app.config import settings


class DocumentEmbedder:
    """Embed document chunks and store them in Qdrant."""

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

    def _ensure_collection(self):
        collections = self.qdrant_client.get_collections().collections
        exists = any(c.name == settings.COLLECTION_NAME for c in collections)

        if not exists:
            self.qdrant_client.create_collection(
                collection_name=settings.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSIONS,
                    distance=Distance.COSINE,
                ),
            )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed texts locally using sentence-transformers."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_and_store(self, chunks: List[Document]) -> int:
        self._ensure_collection()

        texts = [chunk.page_content for chunk in chunks]

        batch_size = 100
        all_points = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]
            embeddings = self._get_embeddings(batch_texts)

            for chunk, embedding in zip(batch_chunks, embeddings):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk.page_content,
                        **chunk.metadata,
                    },
                )
                all_points.append(point)

        self.qdrant_client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=all_points,
        )

        return len(all_points)

    def delete_collection(self):
        self.qdrant_client.delete_collection(
            collection_name=settings.COLLECTION_NAME,
        )