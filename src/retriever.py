"""Hybrid retriever for finance RAG (dense + BM25)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedder import get_embedder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "finance.index"
MAPPING_PATH = DATA_DIR / "finance_mapping.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def minmax_scale(values: np.ndarray) -> np.ndarray:
    """Safely scale scores to [0, 1]."""
    if values.size == 0:
        return values
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    if np.isclose(v_max, v_min):
        return np.zeros_like(values)
    return (values - v_min) / (v_max - v_min)


class HybridRetriever:
    """Dense + BM25 hybrid retriever over finance chunks."""

    def __init__(self, index_path: Path = INDEX_PATH, mapping_path: Path = MAPPING_PATH) -> None:
        self.index = faiss.read_index(str(index_path))
        with mapping_path.open("r", encoding="utf-8") as f:
            self.mapping: List[Dict[str, str]] = json.load(f)

        self.model = get_embedder(MODEL_NAME)
        self.docs = [row["text"] for row in self.mapping]
        self.tokenized_docs = [doc.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def dense_search(self, query: str, top_k: int = 10) -> List[Dict[str, float]]:
        query_vector = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)
        results: List[Dict[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            row = self.mapping[idx]
            results.append({"id": row["id"], "title": row["title"], "text": row["text"], "dense_score": float(score)})
        return results

    def bm25_search(self, query: str) -> np.ndarray:
        query_tokens = query.lower().split()
        return np.array(self.bm25.get_scores(query_tokens), dtype="float32")

    def retrieve(self, query: str, top_k: int = 3, alpha: float = 0.6) -> List[Dict[str, float]]:
        """Return top-k chunks using weighted hybrid scoring."""
        if not query.strip() or not self.mapping:
            return []

        dense_candidates = self.dense_search(query, top_k=max(10, top_k * 3))
        if not dense_candidates:
            return []

        bm25_scores = self.bm25_search(query)

        dense_scores = np.array([item["dense_score"] for item in dense_candidates], dtype="float32")
        dense_norm = minmax_scale(dense_scores)

        bm25_subset = np.array([bm25_scores[int(item["id"])] for item in dense_candidates], dtype="float32")
        bm25_norm = minmax_scale(bm25_subset)

        hybrid = alpha * dense_norm + (1 - alpha) * bm25_norm

        ranked = []
        for item, score, bm in zip(dense_candidates, hybrid, bm25_subset):
            row = dict(item)
            row["bm25_score"] = float(bm)
            row["hybrid_score"] = float(score)
            ranked.append(row)

        ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()
    sample_query = "How do interest rates affect bond prices?"
    results = retriever.retrieve(sample_query, top_k=3)
    for i, item in enumerate(results, start=1):
        print(f"[{i}] {item['title']} | hybrid={item['hybrid_score']:.4f}")
        print(item["text"][:240], "\n")
