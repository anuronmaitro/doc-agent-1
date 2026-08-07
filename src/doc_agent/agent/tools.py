"""Stage 6 — FIXED tool interface — the agent's tools"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..contracts import *  # noqa


class Tool(ABC):
    name: str

    @abstractmethod
    def __call__(self, **kwargs: Any) -> ToolResult: ...


# FIXED tool set — names & signatures locked (test_tools.py checks these).
class Retrieve(Tool):
    name = "retrieve"

    def __call__(self, query: str, k: int = 10) -> ToolResult:  # type: ignore[override]
        # IMPLEMENT: run the retriever; return ToolResult(ok=True, payload={"chunk_ids": [...],
        #   "top_score": <best chunk score>, "k": k}) so decide() and traces/run.jsonl can read evidence strength.
        raise NotImplementedError


class Rerank(Tool):
    name = "rerank"

    def __call__(self, query: str, candidates: list) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError


class ReadPage(Tool):
    name = "read_page"

    def __call__(self, page_id: str) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError


class EnhancePage(Tool):
    name = "enhance_page"

    def __call__(self, page_id: str) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError


class Extract(Tool):
    name = "extract"

    def __call__(self, field: str, chunk_id: str) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError


class Aggregate(Tool):
    name = "aggregate"

    def __call__(self, op: str, items: list) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError


class Cite(Tool):
    name = "cite"

    def __call__(self, chunk_id: str, span: tuple) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError


class Calculator(Tool):
    name = "calculator"

    def __call__(self, expr: str) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError


class EscalateToHuman(Tool):  # HITL entry
    name = "escalate_to_human"

    def __call__(self, reason: str, context: dict) -> ToolResult:  # type: ignore[override]
        raise NotImplementedError


REGISTRY = [
    Retrieve,
    Rerank,
    ReadPage,
    EnhancePage,
    Extract,
    Aggregate,
    Cite,
    Calculator,
    EscalateToHuman,
]
