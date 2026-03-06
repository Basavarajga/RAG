"""Shared embedding utilities with offline fallback support."""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class HashingEmbedder:
    """Deterministic local embedder for offline environments."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def encode(self, texts: List[str], normalize_embeddings: bool = True, convert_to_numpy: bool = True, **_: object) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            for token in tokens:
                digest = hashlib.md5(token.encode("utf-8")).hexdigest()
                idx = int(digest, 16) % self.dim
                matrix[i, idx] += 1.0

        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            matrix = matrix / norms

        return matrix if convert_to_numpy else matrix.tolist()


def get_embedder(model_name: str = MODEL_NAME):
    """Load SentenceTransformer if available, else local deterministic fallback."""
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        print(f"[WARN] Could not load embedding model '{model_name}': {exc}")
        print("[WARN] Falling back to local HashingEmbedder.")
        return HashingEmbedder()
