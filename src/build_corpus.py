"""Build a finance-focused text corpus from local files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.text_processing import (
    extract_text_from_pdf,
    read_text_file,
    split_into_chunks,
    supported_files,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
CORPUS_PATH = DATA_DIR / "finance_corpus.json"


def read_document_text(path: Path) -> str:
    """Load text from one supported document."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".txt":
        return read_text_file(path)
    return ""


def build_corpus(raw_docs_dir: Path = RAW_DOCS_DIR) -> List[Dict[str, str]]:
    """Build chunked corpus records from local TXT/PDF files."""
    corpus: List[Dict[str, str]] = []
    doc_id = 0

    raw_docs_dir.mkdir(parents=True, exist_ok=True)

    for path in supported_files(raw_docs_dir):
        try:
            text = read_document_text(path)
        except Exception as exc:
            print(f"[WARN] Failed to read '{path.name}': {exc}")
            continue

        chunks = split_into_chunks(text)
        if not chunks:
            print(f"[WARN] No chunks produced for: {path.name}")
            continue

        title = path.stem
        for chunk in chunks:
            corpus.append({"id": str(doc_id), "title": title, "text": chunk})
            doc_id += 1

        print(f"[INFO] Added {len(chunks)} chunks from: {path.name}")

    if not corpus:
        print(f"[WARN] No corpus entries built. Add .pdf or .txt files under: {raw_docs_dir}")

    return corpus


def save_corpus(corpus: List[Dict[str, str]], output_path: Path = CORPUS_PATH) -> None:
    """Persist corpus chunks as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)


def main() -> None:
    corpus = build_corpus()
    save_corpus(corpus)
    print(f"[INFO] Saved {len(corpus)} chunks to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
