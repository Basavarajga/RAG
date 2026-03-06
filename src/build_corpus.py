"""Build a finance-focused text corpus from Wikipedia.

This script downloads selected Wikipedia pages, cleans their text, chunks content,
and saves chunks as JSON for downstream embedding and retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import wikipedia


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CORPUS_PATH = DATA_DIR / "finance_corpus.json"

# A compact but broad set of finance-related seed pages.

FALLBACK_DOCUMENTS = {
    "Inflation": "Inflation is the rate at which the general level of prices for goods and services rises over time, reducing purchasing power. Central banks often respond with monetary policy tools such as interest rate changes to stabilize prices.",
    "Bonds and interest rates": "Bond prices and market interest rates move in opposite directions. When interest rates rise, existing bonds with lower coupons become less attractive, so their prices decline. When rates fall, existing bond prices tend to rise.",
    "Dividend investing": "A dividend is a distribution of a company's earnings to shareholders. Investors may evaluate dividend yield, payout ratio, and earnings stability when assessing dividend-focused investment strategies.",
    "Central banking": "A central bank manages monetary policy, often by setting benchmark interest rates and controlling money supply. Its goals commonly include price stability, full employment, and financial system stability.",
}

FINANCE_PAGES = [
    "Finance",
    "Stock market",
    "Bond (finance)",
    "Inflation",
    "Interest rate",
    "Exchange rate",
    "Risk management",
    "Financial statement",
    "Cash flow",
    "Dividend",
    "Asset",
    "Liability (financial accounting)",
    "Equity (finance)",
    "Market capitalization",
    "Price-to-earnings ratio",
    "Central bank",
    "Monetary policy",
    "Fiscal policy",
    "Mutual fund",
    "Exchange-traded fund",
]


def clean_text(text: str) -> str:
    """Normalize Wikipedia content for retrieval quality."""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = text.replace(" ,", ",").replace(" .", ".")
    return text.strip()


def split_into_chunks(
    text: str, chunk_words: int = 240, overlap_words: int = 40, min_words: int = 80
) -> List[str]:
    """Split cleaned text into overlapping chunks of ~200-300 words."""
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


def download_page_content(title: str) -> str:
    """Download one page from Wikipedia, returning empty string on recoverable errors."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.content
    except wikipedia.exceptions.DisambiguationError as exc:
        # Gracefully handle ambiguous pages by trying the first suggested option.
        if exc.options:
            try:
                page = wikipedia.page(exc.options[0], auto_suggest=False)
                return page.content
            except Exception:
                return ""
        return ""
    except wikipedia.exceptions.PageError:
        return ""
    except wikipedia.exceptions.WikipediaException:
        return ""
    except Exception:
        return ""


def build_corpus(page_titles: List[str] | None = None) -> List[Dict[str, str]]:
    """Build chunked corpus records from selected Wikipedia pages."""
    page_titles = page_titles or FINANCE_PAGES

    corpus: List[Dict[str, str]] = []
    doc_id = 0

    for title in page_titles:
        raw_text = download_page_content(title)
        if not raw_text:
            print(f"[WARN] Skipping page due to retrieval issue: {title}")
            continue

        cleaned = clean_text(raw_text)
        chunks = split_into_chunks(cleaned)
        if not chunks:
            print(f"[WARN] No chunks produced for: {title}")
            continue

        for chunk in chunks:
            corpus.append(
                {
                    "id": str(doc_id),
                    "title": title,
                    "text": chunk,
                }
            )
            doc_id += 1

        print(f"[INFO] Added {len(chunks)} chunks from: {title}")

    return corpus


def save_corpus(corpus: List[Dict[str, str]], output_path: Path = CORPUS_PATH) -> None:
    """Persist corpus chunks as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)


def maybe_add_fallback_corpus(corpus: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Use a small offline fallback corpus if Wikipedia is unavailable."""
    if corpus:
        return corpus

    print("[WARN] Wikipedia returned no usable pages. Using fallback finance snippets.")
    doc_id = 0
    for title, text in FALLBACK_DOCUMENTS.items():
        for chunk in split_into_chunks(clean_text(text), chunk_words=220, overlap_words=20, min_words=20):
            corpus.append({"id": str(doc_id), "title": title, "text": chunk})
            doc_id += 1
    return corpus


def main() -> None:
    corpus = build_corpus()
    corpus = maybe_add_fallback_corpus(corpus)
    save_corpus(corpus)
    print(f"[INFO] Saved {len(corpus)} chunks to {CORPUS_PATH}")


if __name__ == "__main__":
    main()
