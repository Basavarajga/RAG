# Finance RAG System

A complete Retrieval-Augmented Generation (RAG) pipeline for financial question answering.

## Architecture

The project follows this flow:

1. **Corpus Builder (`src/build_corpus.py`)**
   - Downloads finance-related Wikipedia pages.
   - Cleans noisy text.
   - Splits text into ~240-word chunks (within the requested 200–300 range).
   - Saves chunks to `data/finance_corpus.json`.

2. **Embeddings + Vector DB (`src/embeddings.py`)**
   - Loads corpus chunks.
   - Generates dense embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
   - Normalizes vectors for cosine similarity.
   - Builds a FAISS inner-product index.
   - Saves:
     - `data/finance_embeddings.npy`
     - `data/finance.index`
     - `data/finance_mapping.json`

3. **Hybrid Retriever (`src/retriever.py`)**
   - Dense retrieval with FAISS.
   - Sparse retrieval with BM25.
   - Hybrid scoring: `alpha * dense + (1-alpha) * bm25`.
   - Returns top-k relevant chunks.

4. **RAG Pipeline (`src/rag_pipeline.py`)**
   - Accepts user query.
   - Retrieves top 3 chunks.
   - Generates an answer with **TinyLlama** (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
   - Falls back gracefully if the model cannot be loaded.

5. **Evaluation (`evaluation/evaluate.py`)**
   - Runs a small financial query test set.
   - Computes and prints retrieval **Precision@k**.

## Project Structure

```text
src/
  build_corpus.py
  embeddings.py
  retriever.py
  rag_pipeline.py
evaluation/
  evaluate.py
README.md
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

1. Build corpus:
```bash
python src/build_corpus.py
```

2. Build embeddings + FAISS index:
```bash
python src/embeddings.py
```

3. Start interactive RAG QA:
```bash
python src/rag_pipeline.py
```

4. Evaluate retrieval quality:
```bash
python evaluation/evaluate.py
```

## Example Queries

- `How do interest rates affect bond prices?`
- `What is the role of a central bank in monetary policy?`
- `How is market capitalization calculated?`
- `What does price-to-earnings ratio indicate?`

## Error Handling

- Wikipedia fetch failures are handled gracefully (pages are skipped with warnings).
- Empty retrieval results return:
  - **"Not found in knowledge base."**
- If LLM loading fails (for example, no model download access), generation falls back to a context-based response.

## Notes

- Designed to run locally end-to-end.
- Main entry point for QA is:
  - `python src/rag_pipeline.py`
