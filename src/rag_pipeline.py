"""End-to-end finance RAG pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever import HybridRetriever


BASE_DIR = Path(__file__).resolve().parents[1]
LLM_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NOT_FOUND_MESSAGE = "Not found in knowledge base."


class FinanceRAG:
    """Finance QA pipeline using hybrid retrieval + lightweight generation."""

    def __init__(self, llm_name: str = LLM_NAME) -> None:
        self.retriever = HybridRetriever()
        self.llm_name = llm_name
        self.tokenizer = None
        self.model = None
        self._load_generation_model()

    def _load_generation_model(self) -> None:
        """Load a lightweight open-source LLM for answer generation."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        except Exception as exc:
            print(f"[WARN] Could not load generation model '{self.llm_name}': {exc}")
            print("[WARN] Falling back to extractive context response.")
            self.tokenizer = None
            self.model = None

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

        if self.model is None or self.tokenizer is None:
            # Graceful fallback when model download is unavailable.
            return f"Based on retrieved context: {contexts[0][:500]}"

        prompt = self._build_prompt(query, contexts)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=160,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("Answer:")[-1].strip()
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
