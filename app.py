from __future__ import annotations

import io
import os
from typing import Dict, List, Optional, Tuple

import faiss
import requests
import streamlit as st

from src.embedder import get_embedder
from src.rag_pipeline import FinanceRAG, build_citations
from src.text_processing import clean_text, split_into_chunks

DEFAULT_API_URL = os.getenv("FINANCE_RAG_API_URL", "http://127.0.0.1:8000/ask")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def get_local_rag() -> FinanceRAG:
    return FinanceRAG()


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return get_embedder(MODEL_NAME)


def ask_via_api(query: str, api_url: str, timeout_s: float = 30.0) -> Tuple[str, List[Dict[str, object]]]:
    response = requests.get(api_url, params={"query": query}, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    return payload.get("answer", "No answer returned by API."), payload.get("citations", [])


def ask_via_local_model(query: str) -> Tuple[str, List[Dict[str, object]]]:
    rag = get_local_rag()
    result = rag.answer_with_citations(query)
    return str(result["answer"]), result["citations"]


def extract_uploaded_pdf_text(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF upload support.") from exc

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return clean_text("\n".join(pages))


def build_uploaded_pdf_index(
    file_bytes: bytes, filename: str
) -> Tuple[faiss.IndexFlatIP, List[Dict[str, object]]]:
    text = extract_uploaded_pdf_text(file_bytes)
    chunks = split_into_chunks(text)
    if not chunks:
        raise ValueError("Could not extract enough text to create chunks from the uploaded PDF.")

    model = get_embedding_model()
    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    rows = [
        {"id": str(i), "title": filename, "chunk_number": i + 1, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]
    return index, rows


def retrieve_from_uploaded_pdf(query: str, top_k: int = 3) -> List[Dict[str, object]]:
    index: Optional[faiss.IndexFlatIP] = st.session_state.get("uploaded_index")
    rows: List[Dict[str, object]] = st.session_state.get("uploaded_chunks", [])
    if index is None or not rows:
        return []

    model = get_embedding_model()
    qvec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(qvec, top_k)

    results: List[Dict[str, object]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0:
            continue
        row = rows[idx]
        results.append({
            "id": row["id"],
            "title": row["title"],
            "chunk_number": row["chunk_number"],
            "text": row["text"],
            "score": float(score),
        })
    return results


def answer_from_uploaded_pdf(query: str) -> Tuple[str, List[Dict[str, object]]]:
    results = retrieve_from_uploaded_pdf(query)
    if not results:
        raise ValueError("No relevant context found in uploaded PDF.")

    # FIX: route through the LLM via generate_answer() instead of raw string slice
    rag = get_local_rag()
    contexts = [r["text"] for r in results]
    answer = rag.generate_answer(query, contexts)
    return answer, results


def run_query(
    query: str, mode: str, api_url: str
) -> tuple[Optional[str], Optional[str], Optional[List[Dict[str, object]]], List[Dict[str, object]]]:
    try:
        if st.session_state.get("uploaded_index") is not None:
            answer, contexts = answer_from_uploaded_pdf(query)
            return answer, None, contexts, build_citations(contexts)

        if mode == "API":
            answer, citations = ask_via_api(query, api_url)
            return answer, None, None, citations

        answer, citations = ask_via_local_model(query)
        return answer, None, None, citations
    except requests.RequestException as exc:
        return None, f"API request failed: {exc}", None, []
    except Exception as exc:
        return None, f"Failed to answer query: {exc}", None, []


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_index" not in st.session_state:
        st.session_state.uploaded_index = None
    if "uploaded_chunks" not in st.session_state:
        st.session_state.uploaded_chunks = []
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None


st.set_page_config(page_title="Finance AI Assistant", page_icon="💰", layout="centered")
init_state()

st.title("💰 Finance AI Assistant")
st.caption("Ask finance questions using either the API or local in-process RAG pipeline.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Answer source",
        ["API", "Local"],
        help="Use API mode if uvicorn is running, or Local mode for direct in-app inference.",
    )
    api_url = st.text_input("API URL", value=DEFAULT_API_URL, disabled=(mode != "API"))

    st.markdown("---")
    st.subheader("Retrieval tuning")
    alpha = st.slider(
        "Hybrid alpha (dense vs BM25)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="1.0 = pure dense, 0.0 = pure BM25",
    )
    st.session_state["hybrid_alpha"] = alpha

    st.markdown("---")
    st.subheader("Upload custom PDF")
    uploaded_pdf = st.file_uploader("Upload a PDF for ad-hoc retrieval", type=["pdf"])

    if uploaded_pdf is not None:
        file_bytes = uploaded_pdf.getvalue()
        if st.session_state.uploaded_filename != uploaded_pdf.name:
            with st.spinner("Indexing uploaded PDF..."):
                try:
                    index, rows = build_uploaded_pdf_index(file_bytes, uploaded_pdf.name)
                    st.session_state.uploaded_index = index
                    st.session_state.uploaded_chunks = rows
                    st.session_state.uploaded_filename = uploaded_pdf.name
                    st.success(f"Indexed {len(rows)} chunks from {uploaded_pdf.name}")
                except Exception as exc:
                    st.session_state.uploaded_index = None
                    st.session_state.uploaded_chunks = []
                    st.session_state.uploaded_filename = None
                    st.error(f"Could not index uploaded PDF: {exc}")

    if st.session_state.uploaded_filename:
        if st.button("Clear uploaded PDF index"):
            st.session_state.uploaded_index = None
            st.session_state.uploaded_chunks = []
            st.session_state.uploaded_filename = None
            st.success("Cleared uploaded PDF index.")

    st.markdown("---")
    st.subheader("Quick examples")
    examples = [
        "How do interest rates affect bond prices?",
        "What does a central bank do in monetary policy?",
        "How is market capitalization calculated?",
    ]
    for idx, example in enumerate(examples):
        if st.button(example, key=f"example_{idx}"):
            st.session_state["query_input"] = example

query = st.text_input(
    "Ask a question",
    key="query_input",
    placeholder="e.g., Why does inflation reduce purchasing power?",
)

col1, col2 = st.columns([1, 1])
with col1:
    submit_clicked = st.button("Submit", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("Clear history", use_container_width=True)

if clear_clicked:
    st.session_state.chat_history = []

if submit_clicked:
    clean_query = query.strip()
    if not clean_query:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            answer, error, contexts, citations = run_query(clean_query, mode, api_url)

        if error:
            st.error(error)
        else:
            response_mode = "Uploaded PDF" if contexts else mode
            st.session_state.chat_history.append(
                {
                    "query": clean_query,
                    "answer": answer,
                    "mode": response_mode,
                    "contexts": contexts or [],
                    "citations": citations,
                }
            )

if st.session_state.chat_history:
    st.markdown("### Conversation")
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {entry['query']}")
        st.markdown(f"**Assistant ({entry['mode']}):** {entry['answer']}")
        if entry.get("citations"):
            st.markdown("**Sources:**")
            for citation in entry["citations"]:
                st.markdown(
                    f"- {citation['pdf_filename']} — Chunk {citation['chunk_number']}"
                )
        if entry.get("contexts"):
            with st.expander("Source context"):
                for i, ctx in enumerate(entry["contexts"], start=1):
                    st.markdown(f"**Chunk {i}** (score={ctx['score']:.3f})")
                    st.write(ctx["text"])
        st.markdown("---")
