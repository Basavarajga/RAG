from __future__ import annotations

from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.rag_pipeline import FinanceRAG

app = FastAPI(title="Finance RAG API")


# FIX: lazy-load FinanceRAG so the heavy model + FAISS index are only
# initialised on the first actual query, not at import/startup time.
@lru_cache(maxsize=1)
def get_rag() -> FinanceRAG:
    return FinanceRAG()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Finance RAG API is running"}


@app.get("/ask")
def ask(query: str = Query(...)):
    answer = get_rag().answer(query)
    return {"query": query, "answer": answer}


@app.post("/ask")
def ask_post(request: QueryRequest):
    answer = get_rag().answer(request.query)
    return {"query": request.query, "answer": answer}
