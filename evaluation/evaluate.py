"""Simple retrieval evaluation for the finance RAG retriever."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# Make src importable when running: python evaluation/evaluate.py
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# FIX: use fully-qualified import consistent with the rest of the codebase
from src.retriever import HybridRetriever  # noqa: E402

EVAL_SET: List[Dict[str, object]] = [
    {
        "query": "What is inflation and how does it affect purchasing power?",
        "keywords": ["inflation", "purchasing power"],
    },
    {
        "query": "How do dividends relate to stock ownership?",
        "keywords": ["dividend", "shareholder"],
    },
    {
        "query": "What does a central bank do in monetary policy?",
        "keywords": ["central bank", "monetary policy"],
    },
    {
        "query": "Why do bond prices change when interest rates change?",
        "keywords": ["bond", "interest rate"],
    },
]


def precision_at_k(results: List[Dict[str, object]], keywords: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = results[:k]
    if not top:
        return 0.0

    hits = 0
    for row in top:
        text = str(row.get("text", "")).lower()
        if any(keyword.lower() in text for keyword in keywords):
            hits += 1
    return hits / k


def evaluate(k: int = 3) -> None:
    retriever = HybridRetriever()

    scores = []
    print(f"Evaluating retrieval with Precision@{k}\n")

    for sample in EVAL_SET:
        query = str(sample["query"])
        keywords = [str(x) for x in sample["keywords"]]
        results = retriever.retrieve(query, top_k=k)

        p_at_k = precision_at_k(results, keywords, k)
        scores.append(p_at_k)

        print(f"Query: {query}")
        print(f"Precision@{k}: {p_at_k:.2f}")
        print("-" * 60)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Average Precision@{k}: {avg_score:.2f}")


if __name__ == "__main__":
    evaluate(k=3)
