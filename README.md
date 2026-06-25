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
   - Retrieves and reranks context chunks.
   - Generates an answer with Groq (`llama-3.1-8b-instant`) when `GROQ_API_KEY` is set.
   - Falls back gracefully to an extractive context response if `GROQ_API_KEY` is unavailable.

5. **FastAPI (`api.py`)**
   - Exposes `/ask` as GET and POST endpoints for programmatic question answering.

6. **Streamlit UI (`app.py`)**
   - Provides an interactive chat interface.
   - Can call the FastAPI service or run the local in-process RAG pipeline.
   - Supports ad-hoc PDF upload and in-session indexing.

7. **Evaluation (`evaluation/evaluate.py`)**
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
data/
  finance_corpus.json
  finance_mapping.json
api.py
app.py
Dockerfile
.dockerignore
README.md
requirements.txt
```

## Configuration

The application reads secrets and runtime settings from environment variables.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | Recommended | unset | Enables Groq-hosted answer generation. If omitted, the app still runs and returns an extractive fallback from retrieved context. |
| `FINANCE_RAG_API_URL` | No | `http://127.0.0.1:8000/ask` | Streamlit API-mode endpoint. The Docker image sets this for the in-container FastAPI service. |
| `API_PORT` | No | `8000` | FastAPI port inside the container command. |
| `STREAMLIT_PORT` | No | `8501` | Streamlit port inside the container command. |

Create a local `.env` file if desired, but do not commit it:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

## Local Execution

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install all dependencies:

```bash
pip install -r requirements.txt
```

3. Export the Groq key for generated answers:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

4. If you have new source documents, add `.pdf` or `.txt` files to `data/raw_docs/` and rebuild the corpus:

```bash
python src/build_corpus.py
```

5. Build or refresh the local FAISS index:

```bash
python src/embeddings.py
```

6. Start the FastAPI service:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

7. In a second terminal, start the Streamlit UI:

```bash
streamlit run app.py
```

8. Open Streamlit at `http://localhost:8501`. In API mode, keep the default API URL `http://127.0.0.1:8000/ask`; in Local mode, Streamlit runs the RAG pipeline in-process.

## Command-Line Usage and Checks

Run a single RAG query from the command line:

```bash
python src/rag_pipeline.py -q "How do interest rates affect bond prices?"
```

Evaluate retrieval quality:

```bash
python evaluation/evaluate.py
```

Run the full project checks, including corpus/index generation and an API smoke test:

```bash
bash scripts/run_full_checks.sh
```

## Docker Deployment

The Docker image installs the dependencies from `requirements.txt`, builds the FAISS index from the checked-in corpus during the image build, and starts both FastAPI and Streamlit in the same container.

1. Build the image:

```bash
docker build -t finance-rag-system .
```

2. Run the container with your Groq API key:

```bash
docker run --rm \
  -p 8000:8000 \
  -p 8501:8501 \
  -e GROQ_API_KEY="your_groq_api_key_here" \
  finance-rag-system
```

3. Open the services:

- Streamlit UI: `http://localhost:8501`
- FastAPI health endpoint: `http://localhost:8000/`
- FastAPI ask endpoint: `http://localhost:8000/ask?query=How%20do%20interest%20rates%20affect%20bond%20prices%3F`

If you need to use different ports inside the container, set `API_PORT` and `STREAMLIT_PORT` and update the published port mappings accordingly.

## Dependency Verification

All runtime dependencies needed by the API, Streamlit UI, retrieval pipeline, PDF upload, Groq generation, embeddings, FAISS index, yfinance helper, and evaluation scripts are listed in `requirements.txt`.
