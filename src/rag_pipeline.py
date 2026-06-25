"""End-to-end finance RAG pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

from groq import Groq

from src.retriever import HybridRetriever


BASE_DIR = Path(__file__).resolve().parents[1]
LLM_NAME = "llama-3.1-8b-instant"
NOT_FOUND_MESSAGE = "Not found in knowledge base."


def build_citations(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Build de-duplicated answer citations from retrieved chunk metadata."""
    citations: List[Dict[str, object]] = []
    seen = set()

    for item in results:
        filename = str(item.get("title") or item.get("filename") or "Unknown PDF")
        chunk_number = item.get("chunk_number")
        if chunk_number is None:
            chunk_id = item.get("id")
            try:
                chunk_number = int(str(chunk_id)) + 1
            except (TypeError, ValueError):
                chunk_number = len(citations) + 1

        key = (filename, chunk_number)
        if key in seen:
            continue

        seen.add(key)
        citations.append({"pdf_filename": filename, "chunk_number": chunk_number})

    return citations


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
            """
            You are an AI financial analyst.

            Instructions:
            - Answer ONLY using the retrieved financial report context.
            - Do not use outside knowledge.
            - If the answer cannot be found in the retrieved context, respond exactly:
              "Not found in knowledge base."
            - Keep answers concise, factual and well structured.
            """
            f"{joined_context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            return NOT_FOUND_MESSAGE

        if self.client is None:
            # Keep Streamlit/FastAPI usable when GROQ_API_KEY is unavailable.
            return f"Based on retrieved context:\n\n{contexts[0][:500]}"

        prompt = self._build_prompt(query, contexts)

        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI financial analyst. "
                            "Answer ONLY using the provided financial report context. "
                            "Do not use outside knowledge. "
                            "If the answer cannot be found in the context, respond exactly: "
                            "'Not found in knowledge base.'"
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=300,
                temperature=0.1,
            )

            answer = response.choices[0].message.content.strip()
            return answer if answer else NOT_FOUND_MESSAGE

        except Exception as e:
            print(f"[ERROR] Groq API: {e}")
            return "Unable to generate an answer at this time."
    
    def answer(self, query: str, top_k: int = 3) -> str:
        return self.answer_with_citations(query, top_k=top_k)["answer"]

    def answer_with_citations(self, query: str, top_k: int = 3) -> Dict[str, object]:
        results = self.retriever.retrieve(query, top_k=top_k)
        if not results:
            return {"answer": NOT_FOUND_MESSAGE, "citations": []}

        contexts = [item["text"] for item in results if item.get("text")]
        if not contexts:
            return {"answer": NOT_FOUND_MESSAGE, "citations": []}

        return {"answer": self.generate_answer(query, contexts), "citations": build_citations(results)}


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
