from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent import RetailAgent

app = FastAPI(title="Retail AI Assistant API", version="1.0.0")


@lru_cache(maxsize=1)
def get_agent() -> RetailAgent:
    """Lazy-load the agent so startup stays fast."""
    return RetailAgent()


class AskRequest(BaseModel):
    question: str = Field(..., examples=["Is organic milk available and what is the return policy?"])


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)


@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "Retail AI Assistant API is running"}


@app.post("/ask", response_model=AskResponse)
def ask_post(request: AskRequest) -> Dict[str, Any]:
    return get_agent().invoke(request.question)
