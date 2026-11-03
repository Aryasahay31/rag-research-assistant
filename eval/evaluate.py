"""
RAGAS Evaluation — measures how good our RAG pipeline actually is.

RAGAS scores 4 dimensions:

1. FAITHFULNESS (0-1): Does the answer stick to the retrieved context?
   High = no hallucination. Low = making stuff up.

2. ANSWER RELEVANCY (0-1): Does the answer address the question?
   High = on-topic. Low = rambling or off-topic.

3. CONTEXT PRECISION (0-1): Are the retrieved chunks actually relevant?
   High = retriever found the right stuff. Low = noisy retrieval.

4. CONTEXT RECALL (0-1): Did we retrieve ALL the relevant chunks?
   High = nothing important was missed. Low = gaps in retrieval.

Usage:
  python -m eval.evaluate

Reads test questions from eval/test_questions.json, runs each
through the pipeline, then scores with RAGAS.
"""

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval import HybridRetriever, CrossEncoderReranker
from app.generation import RAGChain

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


def load_test_questions(path: str = None) -> list:
    """Load test Q&A pairs from JSON file."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "test_questions.json")

    with open(path, "r") as f:
        return json.load(f)


def run_evaluation():
    """Run the full evaluation pipeline."""
    print("Loading test questions...")
    test_data = load_test_questions()

    # Initialize pipeline components
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    rag_chain = RAGChain()

    # Collect results for RAGAS
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(test_data):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i+1}/{len(test_data)}] {question}")

        # Run retrieval
        candidates = retriever.search(question)
        top_chunks = reranker.rerank(question, candidates)

        # Run generation
        answer = rag_chain.generate(
            query=question,
            chunks=top_chunks,
            session_id=f"eval-{i}",
            use_memory=False,
        )

        # Collect for RAGAS
        questions.append(question)
        answers.append(answer)
        contexts.append([chunk["text"] for chunk in top_chunks])
        ground_truths.append(ground_truth)

        print(f"   Answer: {answer[:100]}...")

    # Build RAGAS dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS evaluation
    print("\n\nRunning RAGAS evaluation...")
    results = evaluate(
        eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # Print results
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in results.items():
        print(f"  {metric:25s}: {score:.4f}")
    print("=" * 50)

    # Save results
    results_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(results_path, "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    run_evaluation()