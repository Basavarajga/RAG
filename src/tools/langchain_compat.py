"""Small compatibility layer for LangChain tool imports."""

from __future__ import annotations

try:  # Prefer the real LangChain abstraction when installed.
    from langchain_core.tools import BaseTool  # type: ignore
except Exception:  # pragma: no cover - used only before dependencies are installed.
    class BaseTool:  # type: ignore[no-redef]
        """Minimal BaseTool fallback so the demo remains importable offline."""

        name: str = "tool"
        description: str = ""

        def __call__(self, tool_input: str) -> str:
            return self.run(tool_input)

        def run(self, tool_input: str) -> str:
            return self._run(tool_input)

        def _run(self, tool_input: str) -> str:
            raise NotImplementedError
