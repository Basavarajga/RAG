# Finance RAG System

A complete Retrieval-Augmented Generation (RAG) pipeline for financial question answering using **local documents** and optional **ad-hoc PDF upload** in Streamlit.

## Architecture

1. **Corpus Builder (`src/build_corpus.py`)**
   - Reads local files from `data/raw_docs/`.
   - Supports `.pdf` and `.txt` sources.
   - Extracts + cleans text.
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
  text_processing.py
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

1. Add your documents to `data/raw_docs/` (`.pdf` and/or `.txt`).

2. Build corpus:
```bash
python src/build_corpus.py
```

3. Build embeddings + FAISS index:
```bash
python src/embeddings.py
```

4. Start interactive RAG QA:
```bash
python src/rag_pipeline.py
```

5. Evaluate retrieval quality:
```bash
python evaluation/evaluate.py
```

6. Run the full project checks (pipeline + API smoke test):
```bash
bash scripts/run_full_checks.sh
```

## Streamlit UI

Run the UI:
```bash
streamlit run app.py
```

By default the UI calls `http://127.0.0.1:8000/ask`. In the sidebar you can:
- Switch to **Local** mode to run RAG directly in-process (no API server required).
- Keep **API** mode and customize the API URL.
- Upload a custom **PDF** for in-session retrieval (auto-indexed with embeddings + FAISS).
- Inspect retrieved chunk snippets under **Source context** in the conversation view.
