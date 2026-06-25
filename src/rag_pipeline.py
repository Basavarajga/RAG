"""End-to-end finance RAG pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

from groq import Groq

from src.retriever import HybridRetriever


BASE_DIR = Path(__file__).resolve().parents[1]
LLM_NAME = "llama-3.1-8b-instant"
NOT_FOUND_MESSAGE = "Not found in knowledge base."


class FinanceRAG:
    """Finance QA pipeline using hybrid retrieval + Groq-hosted generation."""

    def __init__(self, llm_name: str = LLM_NAME) -> None:
        self.retriever = HybridRetriever()
        self.llm_name = llm_name
        self.client = None
        self._init_generation_client()

    def _init_generation_client(self) -> None:
        """Initialise the Groq client without loading any local LLM weights."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[WARN] GROQ_API_KEY is not set. Falling back to extractive context response.")
            return

        self.client = Groq(api_key=api_key)

    @staticmethod
    def _build_prompt(query: str, contexts: List[str]) -> str:
        joined_context = "\n\n".join(f"Context {idx+1}: {ctx}" for idx, ctx in enumerate(contexts))
        return (
            "You are a helpful financial assistant. "
            "Answer the user using only the context. "
            "If the context is insufficient, say 'Not found in knowledge base.'\n\n"
            f"{joined_context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            return NOT_FOUND_MESSAGE

        if self.client is None:
            # Keep Streamlit/FastAPI usable when GROQ_API_KEY is unavailable in local dev.
            return f"Based on retrieved context: {contexts[0][:500]}"

        prompt = self._build_prompt(query, contexts)

        # Generation flow: retrieval has already selected the context; only that
        # context plus the user question is sent to Groq for answer generation.
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=160,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        return answer if answer else NOT_FOUND_MESSAGE

    def answer(self, query: str, top_k: int = 3) -> str:
        results = self.retriever.retrieve(query, top_k=top_k)
        if not results:
            return NOT_FOUND_MESSAGE

        contexts = [item["text"] for item in results if item.get("text")]
        if not contexts:
            return NOT_FOUND_MESSAGE

        return self.generate_answer(query, contexts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the finance RAG QA pipeline.")
    parser.add_argument("-q", "--question", help="Single question to answer in non-interactive mode.")
    args = parser.parse_args()

    rag = FinanceRAG()

    if args.question:
        print(f"Answer: {rag.answer(args.question.strip())}")
        return

    print("Finance RAG ready. Type a question (or 'exit').")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        response = rag.answer(query)
        print(f"Answer: {response}")


if __name__ == "__main__":
    main()
