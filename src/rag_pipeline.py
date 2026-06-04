"""End-to-end retail policy RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from groq import Groq
import os
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.build_corpus import build_corpus, save_corpus
from src.embeddings import build_faiss_index, create_embeddings, save_artifacts
from src.retriever import INDEX_PATH, MAPPING_PATH, HybridRetriever


BASE_DIR = Path(__file__).resolve().parents[1]
NOT_FOUND_MESSAGE = "I could not find that in the retail policy knowledge base."


def ensure_policy_index() -> None:
    """Create the retail policy corpus and FAISS artifacts when missing."""
    if INDEX_PATH.exists() and MAPPING_PATH.exists():
        return

    corpus = build_corpus()
    if not corpus:
        raise ValueError("No retail policy documents found under data/raw_docs/policies/.")

    save_corpus(corpus)
    embeddings = create_embeddings([row["text"] for row in corpus])
    index = build_faiss_index(embeddings)
    save_artifacts(corpus, embeddings, index)


class RetailPolicyRAG:
    """Retail policy QA pipeline using the existing FAISS + BM25 hybrid retriever."""

    def __init__(self) -> None:
        ensure_policy_index()
        self.retriever = HybridRetriever()
        try:
            self.client = Groq(
                api_key=os.getenv("GROQ_API_KEY")
            )
            print("[INFO] Groq client initialized")
        except Exception as e:
            print(f"[ERROR] Groq initialization failed: {e}")
            self.client = None
    
    @staticmethod
    def _compose_fallback_extractive_answer(contexts: List[Dict[str, Any]]) -> str:
        """Return the original deterministic summary from retrieved policy chunks."""
        if not contexts:
            return NOT_FOUND_MESSAGE

        top = contexts[0]
        title = top.get("title", "Retail policy")
        text = str(top.get("text", "")).strip()
        if not text:
            return NOT_FOUND_MESSAGE

        sentences = [part.strip() for part in text.replace("\n", " ").split(". ") if part.strip()]
        summary = ". ".join(sentences[:3]).strip()
        if summary and not summary.endswith("."):
            summary += "."

        return f"According to {title}: {summary}"

    def _compose_extractive_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Return a concise grounded answer from retrieved policy chunks."""
        if not contexts:
            return NOT_FOUND_MESSAGE

        if self.client is None:
            return self._compose_fallback_extractive_answer(contexts)
        context = "\n\n".join(
            item.get("text", "")
            for item in contexts[:3]
        )
        prompt = f"""
You are a retail customer support assistant.

Use ONLY the provided context.

Rules:
- Answer in at most 3 concise bullet points.
- Summarize the relevant information.
- Do not copy the context verbatim.
- Keep the answer short and customer-friendly.
- If the answer is not present in the context, say so.

Context:
{context}

Question:
{query}

Answer:
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=100,
            )

            generated_text = response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return self._compose_fallback_extractive_answer(contexts)

        answer = generated_text.strip()

        return answer or self._compose_fallback_extractive_answer(contexts)
    def answer_with_sources(self, query: str, top_k: int = 3, alpha: float = 0.6) -> Dict[str, Any]:
        """Answer a retail policy query with retrieved source chunks."""
        results = self.retriever.retrieve(query, top_k=top_k, alpha=alpha)
        answer = self._compose_extractive_answer(query, results)
        sources = [
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "source": item.get("source"),
                "chunk_id": item.get("chunk_id"),
                "text": item.get("text"),
                "score": item.get("hybrid_score", item.get("dense_score", 0.0)),
                "dense_score": item.get("dense_score"),
                "bm25_score": item.get("bm25_score"),
            }
            for item in results
        ]
        return {"answer": answer, "sources": sources}

    def answer(self, query: str, top_k: int = 3) -> str:
        """Backward-compatible string answer method."""
        return self.answer_with_sources(query, top_k=top_k)["answer"]


# Backward-compatible alias for older imports.
FinanceRAG = RetailPolicyRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retail policy RAG QA pipeline.")
    parser.add_argument("-q", "--question", help="Single question to answer in non-interactive mode.")
    args = parser.parse_args()

    rag = RetailPolicyRAG()

    if args.question:
        print(f"Answer: {rag.answer(args.question.strip())}")
        return

    print("Retail Policy RAG ready. Type a question (or 'exit').")

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
