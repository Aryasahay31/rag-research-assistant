"""
RAG Pipeline Latency Evaluation
Measures end-to-end and per-stage latency across multiple queries.
"""

import time
import json
import statistics
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.config import settings
from app.ingestion.embedder import DocumentEmbedder
from app.retrieval.semantic_search import SemanticSearch
from app.retrieval.bm25_search import BM25Search
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.query_transform import QueryTransformer
from app.generation.llm import RAGChain
from app.cache.redis_cache import RedisCache

# ── Test queries ──────────────────────────────────────────────────
TEST_QUERIES = [
    "What is hybrid search in RAG?",
    "How does cross-encoder reranking work?",
    "What are the benefits of chunking documents?",
    "Explain vector databases and embeddings",
    "What is the role of prompt engineering in RAG?",
]


def measure(func, *args, **kwargs):
    """Run a function and return (result, elapsed_ms)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    return result, elapsed


def run_latency_eval():
    print("=" * 70)
    print("  RAG PIPELINE LATENCY EVALUATION")
    print("=" * 70)

    # ── Initialize components ─────────────────────────────────────
    print("\nInitializing components...")
    embedder = DocumentEmbedder()
    semantic = SemanticSearch()
    bm25 = BM25Search()
    hybrid = HybridRetriever()
    reranker = CrossEncoderReranker()
    query_transformer = QueryTransformer()
    rag_chain = RAGChain()
    cache = RedisCache()

    # ── Stage-level latency storage ───────────────────────────────
    stages = {
        "embedding":   [],
        "semantic":    [],
        "bm25":        [],
        "hybrid":      [],
        "reranking":   [],
        "hyde":        [],
        "generation":  [],
        "cache_miss":  [],
        "cache_hit":   [],
        "end_to_end":  [],
    }

    print(f"\nRunning {len(TEST_QUERIES)} queries...\n")
    print("-" * 70)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\nQuery {i}: \"{query}\"")

        e2e_start = time.perf_counter()

        # 1. Cache check (miss)
        _, cache_miss_ms = measure(cache.get, query)
        stages["cache_miss"].append(cache_miss_ms)

        # 2. Embedding
        _, embed_ms = measure(embedder._get_embeddings, [query])
        stages["embedding"].append(embed_ms)

        # 3. Semantic search
        sem_results, sem_ms = measure(semantic.search, query)
        stages["semantic"].append(sem_ms)

        # 4. BM25 search
        bm25_results, bm25_ms = measure(bm25.search, query)
        stages["bm25"].append(bm25_ms)

        # 5. Hybrid fusion
        hybrid_results, hybrid_ms = measure(hybrid.search, query)
        stages["hybrid"].append(hybrid_ms)

        # 6. Reranking
        if hybrid_results:
            reranked, rerank_ms = measure(reranker.rerank, query, hybrid_results)
        else:
            reranked, rerank_ms = [], 0.0
        stages["reranking"].append(rerank_ms)

        # 7. HyDE transform
        _, hyde_ms = measure(query_transformer.hyde_transform, query)
        stages["hyde"].append(hyde_ms)

        # 8. LLM generation (using RAGChain with chunks)
        chunks_for_gen = reranked[:5] if reranked else []
        answer, gen_ms = measure(
            rag_chain.generate,
            query,
            chunks_for_gen,
            "latency_eval",
            False,  # use_memory=False for clean eval
        )
        stages["generation"].append(gen_ms)

        e2e_ms = (time.perf_counter() - e2e_start) * 1000
        stages["end_to_end"].append(e2e_ms)

        # 9. Cache set + cache hit
        cache.set(query, {"answer": answer})
        _, cache_hit_ms = measure(cache.get, query)
        stages["cache_hit"].append(cache_hit_ms)

        print(f"  Embed: {embed_ms:7.1f}ms | Semantic: {sem_ms:7.1f}ms | "
              f"BM25: {bm25_ms:7.1f}ms | Rerank: {rerank_ms:7.1f}ms | "
              f"LLM: {gen_ms:7.1f}ms | E2E: {e2e_ms:7.1f}ms")

    # ── Summary statistics ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  LATENCY SUMMARY (milliseconds)")
    print("=" * 70)
    print(f"  {'Stage':<20} {'Mean':>8} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 60)

    report = {}
    for stage, times in stages.items():
        if not times:
            continue
        sorted_times = sorted(times)
        n = len(sorted_times)
        p95_idx = min(int(n * 0.95), n - 1)
        mean = statistics.mean(times)
        median = statistics.median(times)
        p95 = sorted_times[p95_idx]
        mn = min(times)
        mx = max(times)

        print(f"  {stage:<20} {mean:>8.1f} {median:>8.1f} {p95:>8.1f} {mn:>8.1f} {mx:>8.1f}")
        report[stage] = {
            "mean_ms": round(mean, 2),
            "median_ms": round(median, 2),
            "p95_ms": round(p95, 2),
            "min_ms": round(mn, 2),
            "max_ms": round(mx, 2),
        }

    # ── Bottleneck analysis ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BOTTLENECK ANALYSIS")
    print("=" * 70)

    avg_e2e = statistics.mean(stages["end_to_end"])
    stage_avgs = {
        "Embedding":  statistics.mean(stages["embedding"]),
        "Semantic":   statistics.mean(stages["semantic"]),
        "BM25":       statistics.mean(stages["bm25"]),
        "Reranking":  statistics.mean(stages["reranking"]),
        "HyDE":       statistics.mean(stages["hyde"]),
        "Generation": statistics.mean(stages["generation"]),
    }

    for name, avg in sorted(stage_avgs.items(), key=lambda x: -x[1]):
        pct = (avg / avg_e2e) * 100
        bar = "#" * int(pct / 2)
        print(f"  {name:<12} {avg:>8.1f}ms  ({pct:>5.1f}%)  {bar}")

    print(f"\n  Cache speedup: cache hit ({statistics.mean(stages['cache_hit']):.1f}ms) "
          f"vs full pipeline ({avg_e2e:.1f}ms) = "
          f"{avg_e2e / max(statistics.mean(stages['cache_hit']), 0.01):.0f}x faster")

    # ── Save report ───────────────────────────────────────────────
    report_path = os.path.join(os.path.dirname(__file__), "eval", "latency_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: eval/latency_report.json")
    print("=" * 70)


if __name__ == "__main__":
    run_latency_eval()