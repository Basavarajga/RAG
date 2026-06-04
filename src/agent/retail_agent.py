"""LangChain-tool retail assistant agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from src.tools import InventoryTool, PolicyTool, StoreInfoTool
from src.tools.langchain_compat import BaseTool


POLICY_KEYWORDS = {
    "return", "returns", "refund", "policy", "membership", "member", "rewards",
    "benefits", "delivery", "pickup", "faq", "coupon", "coupons", "substitution",
}
INVENTORY_KEYWORDS = {
    "available", "availability", "stock", "inventory", "have", "carry", "aisle",
    "milk", "bread", "coke", "eggs", "bananas", "avocados",
}
STORE_KEYWORDS = {
    "store", "hours", "hour", "open", "close", "closes", "closing", "location",
    "address", "phone", "san jose", "san francisco", "oakland",
}


@dataclass
class ToolCallResult:
    """Normalized tool call payload used by the UI and API."""

    tool: str
    answer: str
    sources: List[Dict[str, Any]]


class RetailAgent:
    """Simple interview-demo agent that routes queries to LangChain tools.

    The class intentionally avoids external LLM APIs so the app runs locally. It still uses
    LangChain-style tool abstractions and keeps the previous RAG stack inside PolicyTool.
    """

    def __init__(self) -> None:
        self.tools: Dict[str, BaseTool] = {
            "policy_tool": PolicyTool(),
            "inventory_tool": InventoryTool(),
            "store_info_tool": StoreInfoTool(),
        }

    @staticmethod
    def _tokens(query: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", query.lower()))

    def plan_tools(self, query: str) -> List[str]:
        """Select one or more tools from the user query."""
        normalized = query.lower()
        tokens = self._tokens(query)
        selected: List[str] = []

        if tokens & POLICY_KEYWORDS or any(phrase in normalized for phrase in ["return policy", "delivery policy"]):
            selected.append("policy_tool")
        if tokens & INVENTORY_KEYWORDS or any(item in normalized for item in ["organic milk", "coke"]):
            selected.append("inventory_tool")
        if tokens & STORE_KEYWORDS or any(city in normalized for city in ["san jose", "san francisco"]):
            selected.append("store_info_tool")

        if not selected:
            selected.append("policy_tool")
        return selected

    @staticmethod
    def _parse_tool_payload(raw: str, tool_name: str) -> ToolCallResult:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"tool": tool_name, "answer": raw, "sources": []}
        return ToolCallResult(
            tool=str(payload.get("tool", tool_name)),
            answer=str(payload.get("answer", "")),
            sources=list(payload.get("sources", [])),
        )

    def invoke(self, question: str) -> Dict[str, Any]:
        """Run the agent and return a structured JSON-ready response."""
        clean_question = question.strip()
        if not clean_question:
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "tools_used": [],
                "reasoning_steps": ["No question text was provided."],
            }

        selected_tools = self.plan_tools(clean_question)
        reasoning_steps = [
            f"Inspected the query and selected: {', '.join(selected_tools)}.",
        ]

        tool_results: List[ToolCallResult] = []
        for tool_name in selected_tools:
            tool = self.tools[tool_name]
            reasoning_steps.append(f"Calling {tool_name} with the customer question.")
            raw_result = tool.run(clean_question)
            parsed = self._parse_tool_payload(raw_result, tool_name)
            tool_results.append(parsed)
            reasoning_steps.append(f"{tool_name} returned {len(parsed.sources)} source item(s).")

        answer_parts = [result.answer for result in tool_results if result.answer]
        final_answer = "\n\n".join(answer_parts) if answer_parts else "I could not find an answer."
        sources = [source for result in tool_results for source in result.sources]
        tools_used = [result.tool for result in tool_results]
        reasoning_steps.append("Combined tool outputs into the final customer-facing answer.")

        return {
            "answer": final_answer,
            "sources": sources,
            "tools_used": tools_used,
            "reasoning_steps": reasoning_steps,
        }


_agent: RetailAgent | None = None


def get_retail_agent() -> RetailAgent:
    """Lazily initialize the retail agent for API and Streamlit usage."""
    global _agent
    if _agent is None:
        _agent = RetailAgent()
    return _agent
