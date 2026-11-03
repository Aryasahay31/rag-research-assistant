"""
Cross-Encoder Reranker — the precision filter.

After hybrid search gives us ~20 candidate chunks, the reranker
scores each one individually against the query using a cross-encoder.

WHY a separate reranker?
- Bi-encoders (used in semantic search) encode query and document
  SEPARATELY, then compare vectors. Fast but approximate.
- Cross-encoders encode query AND document TOGETHER, capturing
  fine-grained interactions between them. Slow but accurate.

We use the bi-encoder for recall (cast a wide net of 20 candidates)
and the cross-encoder for precision (pick the best 5).

Model: ms-marco-MiniLM-L-6-v2 — small enough to run on CPU,
trained on MS MARCO (a search relevance dataset from Microsoft).
"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from app.config import settings


class CrossEncoderReranker:
    """Rerank chunks using a cross-encoder model."""

    def __init__(self):
        # Load model once — it stays in memory
        self.model = CrossEncoder(settings.RERANKER_MODEL)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using the cross-encoder.

        Takes (query, chunk) pairs and scores each one.
        Returns the top_k highest-scoring chunks.
        """
        top_k = top_k or settings.RERANK_TOP_K

        if not results:
            return []

        # Create pairs of (query, chunk_text) for the cross-encoder
        pairs = [(query, r["text"]) for r in results]

        # Score all pairs — returns a numpy array of scores
        scores = self.model.predict(pairs)

        # Attach scores to results
        scored_results = []
        for result, score in zip(results, scores):
            scored_results.append({
                **result,
                "rerank_score": float(score),
            })

        # Sort by rerank score (descending) and take top_k
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_results[:top_k]