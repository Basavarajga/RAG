"""LangChain tool for mock retail inventory lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.langchain_compat import BaseTool

BASE_DIR = Path(__file__).resolve().parents[2]
INVENTORY_PATH = BASE_DIR / "data" / "mock_inventory.json"


class InventoryTool(BaseTool):
    """Return stock availability from a local mock inventory dataset."""

    name: str = "inventory_tool"
    description: str = "Use for product availability, stock count, aisle, or inventory questions."

    @staticmethod
    def _load_inventory() -> List[Dict[str, Any]]:
        with INVENTORY_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _find_item(query: str, inventory: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        normalized_query = query.lower()
        exact_matches = [row for row in inventory if row["item"].lower() in normalized_query]
        if exact_matches:
            return max(exact_matches, key=lambda row: len(row["item"]))

        query_tokens = set(normalized_query.replace("?", "").replace(",", "").split())
        best: Optional[Dict[str, Any]] = None
        best_overlap = 0
        for row in inventory:
            item_tokens = set(row["item"].lower().split())
            overlap = len(query_tokens & item_tokens)
            if overlap > best_overlap:
                best = row
                best_overlap = overlap
        return best if best_overlap else None

    def _run(self, query: str) -> str:
        inventory = self._load_inventory()
        item = self._find_item(query, inventory)
        if item is None:
            return json.dumps(
                {
                    "tool": self.name,
                    "answer": "I could not find that item in the mock inventory dataset.",
                    "sources": [{"source": str(INVENTORY_PATH), "available_items": [r["item"] for r in inventory]}],
                },
                ensure_ascii=False,
            )

        answer = (
            f"{item['item'].title()} is available: {item['stock']} {item.get('unit', 'units')} "
            f"currently in stock in the {item.get('aisle', 'unknown')} aisle."
        )
        return json.dumps(
            {"tool": self.name, "answer": answer, "sources": [{"source": str(INVENTORY_PATH), **item}]},
            ensure_ascii=False,
        )
