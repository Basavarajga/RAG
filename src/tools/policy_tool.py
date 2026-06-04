"""LangChain tool for retail policy RAG search."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict

from src.rag_pipeline import RetailPolicyRAG
from src.tools.langchain_compat import BaseTool


@lru_cache(maxsize=1)
def get_policy_rag() -> RetailPolicyRAG:
    return RetailPolicyRAG()


class PolicyTool(BaseTool):
    """Search retail policy documents with the existing hybrid RAG pipeline."""

    name: str = "policy_tool"
    description: str = (
        "Use for questions about return policy, refunds, membership benefits, "
        "delivery policy, pickup policy, account help, digital coupons, and FAQs."
    )

    def _run(self, query: str) -> str:
        result: Dict[str, Any] = get_policy_rag().answer_with_sources(query)
        return json.dumps({"tool": self.name, **result}, ensure_ascii=False)
