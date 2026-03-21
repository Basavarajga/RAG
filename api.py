from fastapi import FastAPI, Query
from pydantic import BaseModel
from src.rag_pipeline import FinanceRAG

app = FastAPI(title="Finance RAG API")

rag = FinanceRAG()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "Finance RAG API is running"}


@app.get("/ask")
def ask(query: str = Query(...)):
    answer = rag.answer(query)
    return {"query": query, "answer": answer}


@app.post("/ask")
def ask_post(request: QueryRequest):
    answer = rag.answer(request.query)
    return {"query": request.query, "answer": answer}
