"""
RAG Pipeline Benchmark — compares 4 retrieval configurations using RAGAS-style metrics.

Configurations:
  1. Baseline     — semantic search only (no BM25, no reranking, no HyDE)
  2. + Hybrid     — semantic + BM25 with RRF
  3. + Reranking  — hybrid + cross-encoder reranking
  4. + HyDE       — hybrid + reranking + HyDE query expansion

Metrics (computed locally, no paid API needed):
  - Context Precision: fraction of retrieved chunks relevant to the answer
  - Context Recall: did retrieval find chunks covering the ground truth
  - Faithfulness: is the generated answer grounded in retrieved chunks
  - Answer Relevancy: semantic similarity between question and answer
  - Latency: end-to-end time per query

Run: python eval/benchmark.py
"""

import json
import os
import sys
import time
import statistics
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentence_transformers import SentenceTransformer
from app.config import settings
from app.retrieval.semantic_search import SemanticSearch
from app.retrieval.bm25_search import BM25Search
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.query_transform import QueryTransformer
from app.generation.llm import RAGChain


# ── Metric Functions ──────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class MetricCalculator:
    """Compute RAGAS-style metrics using local sentence-transformers."""

    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def context_precision(self, question: str, chunks: List[Dict], ground_truth_context: str) -> float:
        """What fraction of retrieved chunks are relevant to the ground truth context."""
        if not chunks:
            return 0.0

        gt_emb = self.model.encode(ground_truth_context)
        relevant = 0
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))
            chunk_emb = self.model.encode(text)
            sim = cosine_sim(gt_emb, chunk_emb)
            if sim > 0.35:  # threshold for relevance
                relevant += 1

        return relevant / len(chunks)

    def context_recall(self, chunks: List[Dict], ground_truth_context: str) -> float:
        """Did retrieval find chunks that cover the ground truth."""
        if not chunks:
            return 0.0

        gt_emb = self.model.encode(ground_truth_context)
        best_sim = 0.0
        for chunk in chunks:
            text = chunk.get("text", chunk.get("content", ""))
            chunk_emb = self.model.encode(text)
            sim = cosine_sim(gt_emb, chunk_emb)
            best_sim = max(best_sim, sim)

        return min(best_sim / 0.6, 1.0)  # normalize: 0.6+ sim = full recall

    def faithfulness(self, answer: str, chunks: List[Dict]) -> float:
        """Is the answer grounded in the retrieved chunks."""
        if not chunks or not answer.strip():
            return 0.0

        sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        chunk_text = " ".join(
            c.get("text", c.get("content", "")).lower() for c in chunks
        )

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "not", "it", "this", "that", "with",
            "from", "by", "as", "be", "has", "have", "had", "do", "does", "did",
            "will", "can", "may", "its", "also", "such", "which", "these", "those",
            "been", "more", "than", "each", "about", "into", "over", "some",
            "according", "document", "source",
        }

        grounded = 0
        for sentence in sentences:
            words = set(sentence.lower().split())
            keywords = words - stop_words
            if not keywords:
                grounded += 1
                continue

            matches = sum(1 for w in keywords if w in chunk_text)
            overlap = matches / len(keywords) if keywords else 0
            if overlap >= 0.25:
                grounded += 1

        return grounded / len(sentences)

    def answer_relevancy(self, question: str, answer: str) -> float:
        """Semantic similarity between question and answer."""
        if not answer.strip():
            return 0.0

        q_emb = self.model.encode(question)
        a_emb = self.model.encode(answer[:500])  # truncate long answers
        sim = cosine_sim(q_emb, a_emb)
        return max(0.0, min(sim, 1.0))


# ── Retrieval Configurations ─────────────────────────────────────

def retrieve_baseline(question: str, semantic: SemanticSearch) -> List[Dict]:
    """Config 1: Semantic search only."""
    return semantic.search(question)[:5]


def retrieve_hybrid(question: str, hybrid: HybridRetriever) -> List[Dict]:
    """Config 2: Semantic + BM25 with RRF."""
    return hybrid.search(question)[:5]


def retrieve_hybrid_reranked(
    question: str,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
) -> List[Dict]:
    """Config 3: Hybrid + cross-encoder reranking."""
    candidates = hybrid.search(question)
    return reranker.rerank(question, candidates)


def retrieve_hybrid_reranked_hyde(
    question: str,
    hybrid: HybridRetriever,
    reranker: CrossEncoderReranker,
    query_transformer: QueryTransformer,
) -> List[Dict]:
    """Config 4: Hybrid + reranking + HyDE."""
    hyde_query = query_transformer.hyde_transform(question)
    candidates = hybrid.search(hyde_query)
    return reranker.rerank(question, candidates)


# ── Main Benchmark ────────────────────────────────────────────────

def run_benchmark():
    print("=" * 80)
    print("  RAG PIPELINE BENCHMARK")
    print("  Comparing: Baseline vs Hybrid vs +Reranking vs +HyDE")
    print("=" * 80)

    # Load test set
    test_path = os.path.join(os.path.dirname(__file__), "test_questions.json")
    with open(test_path) as f:
        test_set = json.load(f)

    print(f"\n  Test set: {len(test_set)} questions")
    print("  Initializing components...\n")

    # Init components
    semantic = SemanticSearch()
    hybrid = HybridRetriever()
    reranker = CrossEncoderReranker()
    query_transformer = QueryTransformer()
    rag_chain = RAGChain()
    metrics = MetricCalculator()

    configs = {
        "Baseline (semantic only)": lambda q: retrieve_baseline(q, semantic),
        "+ Hybrid (BM25 + RRF)":    lambda q: retrieve_hybrid(q, hybrid),
        "+ Reranking":              lambda q: retrieve_hybrid_reranked(q, hybrid, reranker),
        "+ HyDE":                   lambda q: retrieve_hybrid_reranked_hyde(q, hybrid, reranker, query_transformer),
    }

    all_results = {}

    for config_name, retrieve_fn in configs.items():
        print(f"\n{'─' * 80}")
        print(f"  Running: {config_name}")
        print(f"{'─' * 80}")

        config_metrics = {
            "context_precision": [],
            "context_recall": [],
            "faithfulness": [],
            "answer_relevancy": [],
            "latency_ms": [],
        }

        for i, item in enumerate(test_set, 1):
            question = item["question"]
            gt_context = item["ground_truth_context"]
            print(f"  [{i}/{len(test_set)}] {question[:60]}...", end=" ", flush=True)

            t0 = time.perf_counter()

            # Retrieve
            chunks = retrieve_fn(question)

            # Generate
            answer = rag_chain.generate(
                query=question,
                chunks=chunks,
                session_id=f"bench_{config_name}",
                use_memory=False,
            )

            latency = (time.perf_counter() - t0) * 1000

            # Compute metrics
            cp = metrics.context_precision(question, chunks, gt_context)
            cr = metrics.context_recall(chunks, gt_context)
            faith = metrics.faithfulness(answer, chunks)
            ar = metrics.answer_relevancy(question, answer)

            config_metrics["context_precision"].append(cp)
            config_metrics["context_recall"].append(cr)
            config_metrics["faithfulness"].append(faith)
            config_metrics["answer_relevancy"].append(ar)
            config_metrics["latency_ms"].append(latency)

            print(f"CP={cp:.2f} CR={cr:.2f} F={faith:.2f} AR={ar:.2f} ({latency:.0f}ms)")

        # Average metrics
        all_results[config_name] = {
            k: round(statistics.mean(v), 3) for k, v in config_metrics.items()
        }

    # ── Comparison Table ──────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  BENCHMARK RESULTS — COMPARISON TABLE")
    print("=" * 80)
    print(f"\n  {'Configuration':<30} {'Ctx Prec':>10} {'Ctx Recall':>10} "
          f"{'Faithful':>10} {'Ans Rel':>10} {'Latency':>10}")
    print("  " + "─" * 80)

    for config_name, avg in all_results.items():
        latency_s = avg["latency_ms"] / 1000
        print(f"  {config_name:<30} {avg['context_precision']:>10.3f} "
              f"{avg['context_recall']:>10.3f} {avg['faithfulness']:>10.3f} "
              f"{avg['answer_relevancy']:>10.3f} {latency_s:>9.1f}s")

    # ── Improvement over baseline ─────────────────────────────────
    baseline = all_results["Baseline (semantic only)"]
    best = all_results["+ Reranking"]  # typically the best without HyDE latency cost

    print(f"\n  Improvement (Baseline → +Reranking):")
    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        base_val = baseline[metric]
        best_val = best[metric]
        if base_val > 0:
            pct = ((best_val - base_val) / base_val) * 100
            arrow = "+" if pct >= 0 else ""
            print(f"    {metric:<25} {base_val:.3f} → {best_val:.3f}  ({arrow}{pct:.1f}%)")

    # ── Save results ──────────────────────────────────────────────
    report = {
        "test_set_size": len(test_set),
        "results": all_results,
    }
    report_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to: eval/benchmark_results.json")

    # ── Markdown table for README ─────────────────────────────────
    md_path = os.path.join(os.path.dirname(__file__), "benchmark_table.md")
    with open(md_path, "w") as f:
        f.write("## Benchmark Results\n\n")
        f.write(f"Evaluated on {len(test_set)} questions from RAG_Study_Notes.pdf\n\n")
        f.write("| Configuration | Context Precision | Context Recall | Faithfulness | Answer Relevancy | Avg Latency |\n")
        f.write("|---|---|---|---|---|---|\n")
        for config_name, avg in all_results.items():
            latency_s = avg["latency_ms"] / 1000
            f.write(f"| {config_name} | {avg['context_precision']:.3f} | "
                    f"{avg['context_recall']:.3f} | {avg['faithfulness']:.3f} | "
                    f"{avg['answer_relevancy']:.3f} | {latency_s:.1f}s |\n")
        f.write(f"\nMetrics computed locally using sentence-transformers (no paid API).\n")

    print(f"  Markdown table saved to: eval/benchmark_table.md")
    print(f"  (Copy this into your README.md)")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()