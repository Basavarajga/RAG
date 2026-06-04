"""Embedding and FAISS index builder for the retail policy corpus."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedder import get_embedder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CORPUS_PATH = DATA_DIR / "retail_policy_corpus.json"
EMBEDDINGS_PATH = DATA_DIR / "retail_policy_embeddings.npy"
INDEX_PATH = DATA_DIR / "retail_policy.index"
MAPPING_PATH = DATA_DIR / "retail_policy_mapping.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus(path: Path = CORPUS_PATH) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings(texts: List[str], model_name: str = MODEL_NAME) -> np.ndarray:
    model = get_embedder(model_name)
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vectors.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build cosine-similarity FAISS index with normalized vectors."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_artifacts(corpus: List[Dict[str, str]], embeddings: np.ndarray, index: faiss.Index) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    with MAPPING_PATH.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)


def load_index(index_path: Path = INDEX_PATH, mapping_path: Path = MAPPING_PATH) -> Tuple[faiss.Index, List[Dict[str, str]]]:
    index = faiss.read_index(str(index_path))
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return index, mapping


def main() -> None:
    corpus = load_corpus()
    if not corpus:
        raise ValueError("Corpus is empty. Run src/build_corpus.py first.")

    texts = [row["text"] for row in corpus]
    embeddings = create_embeddings(texts)
    index = build_faiss_index(embeddings)
    save_artifacts(corpus, embeddings, index)

    print(f"[INFO] Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"[INFO] Saved FAISS index: {INDEX_PATH}")
    print(f"[INFO] Saved mapping: {MAPPING_PATH}")


if __name__ == "__main__":
    main()
