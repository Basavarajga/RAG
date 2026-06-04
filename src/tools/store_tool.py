"""LangChain tool for mock retail store information lookup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.tools.langchain_compat import BaseTool

BASE_DIR = Path(__file__).resolve().parents[2]
STORES_PATH = BASE_DIR / "data" / "mock_stores.json"
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class StoreInfoTool(BaseTool):
    """Return store hours and location details from a local mock store dataset."""

    name: str = "store_info_tool"
    description: str = "Use for store hours, opening time, closing time, address, phone, or location questions."

    @staticmethod
    def _load_stores() -> List[Dict[str, Any]]:
        with STORES_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _find_store(query: str, stores: List[Dict[str, Any]]) -> Dict[str, Any]:
        normalized_query = query.lower()
        for store in stores:
            if store["city"].lower() in normalized_query or store["name"].lower() in normalized_query:
                return store
        return stores[0]

    @staticmethod
    def _requested_day(query: str) -> Optional[str]:
        normalized_query = query.lower()
        for day in DAYS:
            if day in normalized_query:
                return day
        return None

    def _run(self, query: str) -> str:
        stores = self._load_stores()
        store = self._find_store(query, stores)
        day = self._requested_day(query)
        hours = store["hours"]

        if day:
            hours_text = f"On {day.title()}, {store['name']} is open {hours[day]}."
        else:
            hours_text = f"Typical hours for {store['name']} are Monday-Thursday {hours['monday']}, Friday-Saturday {hours['friday']}, and Sunday {hours['sunday']}."

        if "close" in query.lower() and day:
            closing_time = hours[day].split(" - ")[-1]
            hours_text = f"{store['name']} closes at {closing_time} on {day.title()}."
        elif "close" in query.lower():
            closing_time = hours["monday"].split(" - ")[-1]
            hours_text = f"{store['name']} usually closes at {closing_time} Monday through Thursday; weekend hours may differ."

        answer = f"{hours_text} Address: {store['address']}. Phone: {store['phone']}."
        return json.dumps(
            {"tool": self.name, "answer": answer, "sources": [{"source": str(STORES_PATH), **store}]},
            ensure_ascii=False,
        )
