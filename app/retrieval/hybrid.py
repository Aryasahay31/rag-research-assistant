"""
Hybrid Retriever — merges semantic and BM25 results using RRF.

Reciprocal Rank Fusion (RRF) is a simple but powerful technique:

    RRF_score(doc) = sum( 1 / (k + rank_in_list) )

where k=60 is a constant that prevents top-ranked docs from
dominating too heavily.

WHY RRF instead of just averaging scores?
- Semantic scores (cosine similarity) range 0-1
- BM25 scores are unbounded (could be 0-50+)
- You can't average apples and oranges
- RRF only uses RANK positions, making it score-agnostic

Example: A chunk ranked #1 in semantic and #5 in BM25:
  RRF = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

A chunk ranked #3 in both:
  RRF = 1/(60+3) + 1/(60+3) = 0.0159 + 0.0159 = 0.0317

So being #1 in one list beats being #3 in both — which makes
intuitive sense (strong signal from one method is valuable).
"""

from typing import List, Dict, Any
from app.retrieval.semantic_search import SemanticSearch
from app.retrieval.bm25_search import BM25Search


class HybridRetriever:
    """Combines semantic and BM25 search using Reciprocal Rank Fusion."""

    def __init__(self, rrf_k: int = 60):
        self.semantic = SemanticSearch()
        self.bm25 = BM25Search()
        self.rrf_k = rrf_k  # RRF constant

    def search(
        self,
        query: str,
        semantic_top_k: int = None,
        bm25_top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Run both searches and merge results with RRF.
        Returns all unique chunks sorted by fused score.
        """
        # Run both searches
        semantic_results = self.semantic.search(query, top_k=semantic_top_k)
        bm25_results = self.bm25.search(query, top_k=bm25_top_k)

        # Build RRF scores
        # Key = chunk text (using text as ID since we don't have UUIDs here)
        rrf_scores: Dict[str, float] = {}
        result_map: Dict[str, Dict[str, Any]] = {}

        # Score from semantic search (rank starts at 1)
        for rank, result in enumerate(semantic_results, start=1):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (self.rrf_k + rank)
            result_map[text] = result  # store full result for output

        # Score from BM25 search
        for rank, result in enumerate(bm25_results, start=1):
            text = result["text"]
            rrf_scores[text] = rrf_scores.get(text, 0) + 1 / (self.rrf_k + rank)
            if text not in result_map:
                result_map[text] = result

        # Sort by RRF score (descending)
        sorted_texts = sorted(
            rrf_scores.keys(),
            key=lambda t: rrf_scores[t],
            reverse=True,
        )

        return [
            {
                "text": text,
                "score": rrf_scores[text],
                "metadata": result_map[text]["metadata"],
            }
            for text in sorted_texts
        ]

    def refresh_bm25(self):
        """Refresh BM25 index after new documents are added."""
        self.bm25.refresh_index()