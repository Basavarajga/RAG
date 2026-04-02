"""Reusable text extraction and chunking utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List



TEXT_EXTENSIONS = {".txt"}
PDF_EXTENSIONS = {".pdf"}


def clean_text(text: str) -> str:
    """Normalize raw text for retrieval/indexing quality."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text.strip()


def split_into_chunks(
    text: str,
    chunk_words: int = 240,
    overlap_words: int = 40,
    min_words: int = 80,
) -> List[str]:
    """Split cleaned text into overlapping chunks (~200-300 words)."""
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    step = max(1, chunk_words - overlap_words)

    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_words]
        if len(chunk) < min_words:
            continue
        chunks.append(" ".join(chunk))
        if start + chunk_words >= len(words):
            break

    return chunks


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Read all extractable text from a PDF file path."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF ingestion. Install dependencies from requirements.txt.") from exc

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def read_text_file(path: Path) -> str:
    return clean_text(path.read_text(encoding="utf-8", errors="ignore"))


def supported_files(directory: Path) -> Iterable[Path]:
    """Yield supported local document files in deterministic order."""
    for path in sorted(directory.glob("*")):
        if path.is_file() and path.suffix.lower() in (TEXT_EXTENSIONS | PDF_EXTENSIONS):
            yield path
