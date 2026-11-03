"""
BM25 Search — the 'sparse' retrieval in our hybrid search.

BM25 (Best Matching 25) is a ranking function used by search engines.
It scores documents based on term frequency (TF) and inverse document
frequency (IDF):

- TF: words that appear more in a chunk score higher
- IDF: rare words score higher than common ones ("kubernetes" beats "the")
- Length normalization: long chunks don't get unfair advantage

WHY combine with semantic search?
Semantic search is great at understanding meaning but can miss exact
keywords. BM25 is great at exact matching but misses synonyms.
Together, they cover each other's blind spots.

Example where BM25 wins: Query "error code 404" — semantic search
might return chunks about "HTTP errors" in general, but BM25 will
find the chunk that literally says "404".
"""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from app.config import settings


class BM25Search:
    """Keyword-based search using BM25 algorithm."""

    def __init__(self):
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.corpus = []        # list of chunk texts
        self.corpus_meta = []   # parallel list of metadata
        self.bm25 = None        # BM25 index, built on first search

    def _load_corpus(self):
        """
        Pull all documents from Qdrant to build the BM25 index.

        BM25 needs all documents in memory to calculate IDF scores.
        This is fine for our scale (thousands of chunks) but wouldn't
        work for millions — then you'd use Elasticsearch.
        """
        # Scroll through all points in the collection
        all_points = []
        offset = None

        while True:
            results, offset = self.qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # we don't need vectors for BM25
            )
            all_points.extend(results)
            if offset is None:
                break

        self.corpus = [point.payload["text"] for point in all_points]
        self.corpus_meta = [
            {k: v for k, v in point.payload.items() if k != "text"}
            for point in all_points
        ]

        # Tokenize: simple whitespace split + lowercase
        # Production systems would use proper tokenization
        tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Score all chunks against the query using BM25.
        Returns top_k results sorted by BM25 score.
        """
        top_k = top_k or settings.BM25_TOP_K

        # Build index on first call (lazy loading)
        if self.bm25 is None:
            self._load_corpus()

        # Tokenize query the same way we tokenized the corpus
        query_tokens = query.lower().split()

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Get top_k indices sorted by score (descending)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {
                "text": self.corpus[i],
                "score": float(scores[i]),
                "metadata": self.corpus_meta[i],
            }
            for i in top_indices
            if scores[i] > 0  # skip zero-score results
        ]

    def refresh_index(self):
        """Force rebuild of BM25 index (call after adding new docs)."""
        self.bm25 = None
        self.corpus = []
        self.corpus_meta = []