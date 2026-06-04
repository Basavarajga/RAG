from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

from src.agent import RetailAgent

DEFAULT_API_URL = os.getenv("RETAIL_AI_API_URL", "http://127.0.0.1:8000/ask")


@st.cache_resource(show_spinner=False)
def get_local_agent() -> RetailAgent:
    return RetailAgent()


def ask_via_api(question: str, api_url: str, timeout_s: float = 45.0) -> Dict[str, Any]:
    response = requests.post(api_url, json={"question": question}, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def ask_via_local_agent(question: str) -> Dict[str, Any]:
    return get_local_agent().invoke(question)


def run_query(question: str, mode: str, api_url: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        if mode == "API":
            return ask_via_api(question, api_url), None
        return ask_via_local_agent(question), None
    except requests.RequestException as exc:
        return None, f"API request failed: {exc}"
    except Exception as exc:
        return None, f"Failed to answer query: {exc}"


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        st.caption("No source chunks or mock records were returned.")
        return

    for idx, source in enumerate(sources, start=1):
        label = source.get("title") or source.get("item") or source.get("name") or f"Source {idx}"
        score = source.get("score")
        score_text = f" · score={score:.3f}" if isinstance(score, (int, float)) else ""
        with st.expander(f"{idx}. {label}{score_text}"):
            if source.get("source"):
                st.caption(f"Source file: {source['source']}")
            if source.get("text"):
                st.write(source["text"])
            else:
                st.json(source)


st.set_page_config(page_title="Retail AI Assistant", page_icon="🛒", layout="wide")
init_state()

st.title("🛒 Retail AI Assistant")
st.caption("Albertsons-style customer support demo with LangChain tools, mock retail data, and FAISS + BM25 policy RAG.")

with st.sidebar:
    st.header("Assistant Settings")
    mode = st.radio(
        "Execution mode",
        ["API", "Local"],
        help="Use API mode with `uvicorn api:app --reload`, or Local mode to run the agent inside Streamlit.",
    )
    api_url = st.text_input("API URL", value=DEFAULT_API_URL, disabled=(mode != "API"))

    st.markdown("---")
    st.subheader("Demo examples")
    examples = [
        "What is the return policy?",
        "Is organic milk available?",
        "What time does the San Jose store close?",
        "Is organic milk available and what is the return policy?",
    ]
    for idx, example in enumerate(examples):
        if st.button(example, key=f"example_{idx}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": example})
            with st.spinner("Routing to retail tools..."):
                payload, error = run_query(example, mode, api_url)
            if error:
                st.session_state.messages.append({"role": "assistant", "content": error, "error": True})
            else:
                st.session_state.messages.append({"role": "assistant", "content": payload["answer"], "payload": payload})
            st.rerun()

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        payload = message.get("payload")
        if payload:
            tools_used = payload.get("tools_used", [])
            if tools_used:
                st.markdown("**Tools used:** " + ", ".join(f"`{tool}`" for tool in tools_used))

            with st.expander("Agent reasoning steps", expanded=False):
                for step in payload.get("reasoning_steps", []):
                    st.markdown(f"- {step}")

            with st.expander("Retrieved source chunks and mock records", expanded=False):
                render_sources(payload.get("sources", []))

if question := st.chat_input("Ask about returns, membership, delivery, inventory, or store hours..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Inspecting query and selecting tools..."):
            payload, error = run_query(question, mode, api_url)
        if error:
            st.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "error": True})
        else:
            st.markdown(payload["answer"])
            tools_used = payload.get("tools_used", [])
            if tools_used:
                st.markdown("**Tools used:** " + ", ".join(f"`{tool}`" for tool in tools_used))
            with st.expander("Agent reasoning steps", expanded=False):
                for step in payload.get("reasoning_steps", []):
                    st.markdown(f"- {step}")
            with st.expander("Retrieved source chunks and mock records", expanded=False):
                render_sources(payload.get("sources", []))
            st.session_state.messages.append({"role": "assistant", "content": payload["answer"], "payload": payload})
