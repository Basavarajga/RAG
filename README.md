# Retail AI Assistant RAG Demo

An Albertsons-style retail customer support assistant that preserves the original local RAG pipeline while adding a simple LangChain-tool agent architecture.

## What it answers

- **Policy questions**: return policy, refunds, membership benefits, delivery and pickup policy, coupon and FAQ support.
- **Inventory questions**: mock stock availability for demo products such as organic milk, bread, coke, eggs, bananas, and avocados.
- **Store questions**: mock store hours, closing times, addresses, and phone numbers for San Jose, San Francisco, and Oakland stores.
- **Multi-tool questions**: the agent can call more than one tool, for example inventory plus policy in one response.

## Architecture

1. **Retail policy corpus builder (`src/build_corpus.py`)**
   - Reads `.txt` and `.pdf` files from `data/raw_docs/policies/`.
   - Cleans text and creates overlapping chunks.
   - Saves `data/retail_policy_corpus.json`.

2. **Embeddings + FAISS (`src/embeddings.py`)**
   - Uses `sentence-transformers/all-MiniLM-L6-v2` when available.
   - Falls back to a deterministic local hashing embedder if the model cannot be loaded.
   - Saves `data/retail_policy_embeddings.npy`, `data/retail_policy.index`, and `data/retail_policy_mapping.json`.

3. **Hybrid retrieval (`src/retriever.py`)**
   - Keeps the existing FAISS dense retrieval plus BM25 sparse retrieval implementation.
   - Combines scores with weighted hybrid scoring.

4. **Policy RAG (`src/rag_pipeline.py`)**
   - Wraps the hybrid retriever for retail policy answers.
   - Auto-builds the small local policy index on first use if artifacts are missing.
   - Returns structured answers with source chunks.

5. **LangChain tools (`src/tools/`)**
   - `PolicyTool`: uses the existing RAG pipeline for retail policy documents.
   - `InventoryTool`: looks up `data/mock_inventory.json`.
   - `StoreInfoTool`: looks up `data/mock_stores.json`.

6. **Agent (`src/agent/retail_agent.py`)**
   - Inspects the user query.
   - Selects one or more LangChain-style tools.
   - Calls tools and combines their structured JSON outputs.
   - Returns `answer`, `sources`, `tools_used`, and `reasoning_steps`.

7. **API and UI**
   - FastAPI: `POST /ask` with `{ "question": "..." }`.
   - Streamlit: retail-branded chat interface with tools used, reasoning steps, and source chunks/mock records.

## Project Structure

```text
api.py
app.py
src/
  agent/
    retail_agent.py
  tools/
    policy_tool.py
    inventory_tool.py
    store_tool.py
  build_corpus.py
  embeddings.py
  retriever.py
  rag_pipeline.py
  text_processing.py
data/
  raw_docs/policies/
    returns.txt
    membership.txt
    delivery.txt
    general_faq.txt
  mock_inventory.json
  mock_stores.json
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build / Refresh the Policy Index

The policy index auto-builds on first query. To build it manually:

```bash
python src/build_corpus.py
python src/embeddings.py
```

## Run Locally

Start the API:

```bash
uvicorn api:app --reload
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Is organic milk available and what is the return policy?"}'
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Example Questions

- `What is the return policy?`
- `Is organic milk available?`
- `What time does the San Jose store close?`
- `Is organic milk available and what is the return policy?`

## Notes for Interview Demo

- No external databases.
- No cloud services.
- No authentication.
- Mock datasets are local JSON files.
- The app remains deterministic and lightweight enough for local demos.
