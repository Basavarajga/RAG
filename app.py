from __future__ import annotations

import os
from typing import Optional

import requests
import streamlit as st

from src.rag_pipeline import FinanceRAG

DEFAULT_API_URL = os.getenv("FINANCE_RAG_API_URL", "http://127.0.0.1:8000/ask")


@st.cache_resource(show_spinner=False)
def get_local_rag() -> FinanceRAG:
    return FinanceRAG()


def ask_via_api(query: str, api_url: str, timeout_s: float = 30.0) -> str:
    response = requests.get(api_url, params={"query": query}, timeout=timeout_s)
    response.raise_for_status()
    payload = response.json()
    return payload.get("answer", "No answer returned by API.")


def ask_via_local_model(query: str) -> str:
    rag = get_local_rag()
    return rag.answer(query)


def run_query(query: str, mode: str, api_url: str) -> tuple[Optional[str], Optional[str]]:
    try:
        if mode == "API":
            return ask_via_api(query, api_url), None
        return ask_via_local_model(query), None
    except requests.RequestException as exc:
        return None, f"API request failed: {exc}"
    except Exception as exc:  # keep UI resilient for demo usage
        return None, f"Failed to answer query: {exc}"


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


st.set_page_config(page_title="Finance AI Assistant", page_icon="💰", layout="centered")
init_state()

st.title("💰 Finance AI Assistant")
st.caption("Ask finance questions using either the API or local in-process RAG pipeline.")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Answer source", ["API", "Local"], help="Use API mode if uvicorn is running, or Local mode for direct in-app inference.")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL, disabled=(mode != "API"))
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

query = st.text_input("Ask a question", key="query_input", placeholder="e.g., Why does inflation reduce purchasing power?")

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
            answer, error = run_query(clean_query, mode, api_url)

        if error:
            st.error(error)
        else:
            st.session_state.chat_history.append({"query": clean_query, "answer": answer, "mode": mode})

if st.session_state.chat_history:
    st.markdown("### Conversation")
    for entry in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {entry['query']}")
        st.markdown(f"**Assistant ({entry['mode']}):** {entry['answer']}")
        st.markdown("---")
