"""Stage 6 — FIXED tool interface — the agent's tools"""

from __future__ import annotations

import ast
import math
import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import config
from ..contracts import *  # noqa
from ..eval.metrics import extract_formulas
from ..index import store
from ..retrieval import rerank as rerank_mod
from ..retrieval import retriever as retriever_mod
from . import hitl


class Tool(ABC):
    name: str

    @abstractmethod
    def __call__(self, **kwargs: Any) -> ToolResult: ...


# --- shared lazy lookups -----------------------------------------------------------------
#
# Small, module-local caches -- deliberately NOT imported from eval/metrics.py, which needs
# the identical chunk_id -> Chunk lookup for citation_accuracy/groundedness. The two files
# stay self-contained (this project's stub convention: each file owns its own body, and
# index/store.py -- the one true shared owner of "how to load the index" -- is an A2 file
# outside Step 7's declared scope) at the cost of ~8 duplicated lines. Same monkeypatch-the-
# loader test pattern as metrics.py's twin and retriever.py's `_ImageIndex._ensure_clip`.

_CHUNK_LOOKUP_CACHE: dict[str, Chunk] | None = None


def _get_chunk_lookup() -> dict[str, Chunk]:
    global _CHUNK_LOOKUP_CACHE
    if _CHUNK_LOOKUP_CACHE is None:
        loaded = store.load(config.load())
        _CHUNK_LOOKUP_CACHE = {c.id: c for c in loaded.chunks}
    return _CHUNK_LOOKUP_CACHE


def _lookup_chunk(chunk_id: str) -> Chunk | None:
    return _get_chunk_lookup().get(chunk_id)


_RETRIEVER_CACHE: retriever_mod.Retriever | None = None


def _get_retriever() -> retriever_mod.Retriever:
    global _RETRIEVER_CACHE
    if _RETRIEVER_CACHE is None:
        _RETRIEVER_CACHE = retriever_mod.Retriever(config.load())
    return _RETRIEVER_CACHE


OCR_DIR = Path("data/ocr")


# --- FIXED tool set — names & signatures locked (test_tools.py checks these). --------------
#
# Every tool below returns ToolResult(ok=..., payload=...) and never raises: bad input gets
# ok=False with a "reason" in payload, not an exception. One tool call raising would crash
# the whole agent turn mid-eval (plan_a3.md Step 7) -- there is no partial credit for a run
# that never produced a trace. Anticipated failures (missing file, unknown chunk, bad span,
# unparseable expression) get a specific reason; a broad except around the one real external
# call each tool makes (retriever/rerank/hitl/file IO) is the last-resort net underneath
# that, not a substitute for it.


class Retrieve(Tool):
    name = "retrieve"

    def __call__(self, query: str, k: int = 10) -> ToolResult:  # type: ignore[override]
        if not query or not query.strip():
            return ToolResult(ok=False, payload={"reason": "query must be non-empty"})
        try:
            chunks = _get_retriever().retrieve(query, k=k)
        except Exception as exc:  # noqa: BLE001 -- a failing retriever must not crash the run
            return ToolResult(ok=False, payload={"reason": f"retrieval failed: {exc}"})
        return ToolResult(
            ok=True,
            payload={
                "chunk_ids": [c.id for c in chunks],
                "top_score": retriever_mod.top_score(chunks),
                "k": k,
            },
        )


class Rerank(Tool):
    name = "rerank"

    def __call__(self, query: str, candidates: list) -> ToolResult:  # type: ignore[override]
        if not candidates:
            return ToolResult(ok=True, payload={"chunk_ids": [], "score_gap": 0.0})
        resolved: list[Chunk] = []
        for c in candidates:
            if isinstance(c, Chunk):
                resolved.append(c)
                continue
            chunk = _lookup_chunk(c)
            if chunk is None:
                return ToolResult(ok=False, payload={"reason": f"no such chunk: {c!r}"})
            resolved.append(chunk)
        try:
            reranked = rerank_mod.rerank(query, resolved, config.load())
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, payload={"reason": f"rerank failed: {exc}"})
        # Top-1 vs top-2 score gap (Step 3 / Explainable NFR) -- rerank.rerank()'s own
        # docstring says this is exactly results[0].score - results[1].score, not a
        # separate return value, so that's what the caller (this tool) computes.
        gap = reranked[0].score - reranked[1].score if len(reranked) >= 2 else 0.0
        return ToolResult(
            ok=True, payload={"chunk_ids": [c.id for c in reranked], "score_gap": gap}
        )


class ReadPage(Tool):
    name = "read_page"

    def __call__(self, page_id: str) -> ToolResult:  # type: ignore[override]
        if not page_id:
            return ToolResult(ok=False, payload={"reason": "page_id must be non-empty"})
        try:
            text = (OCR_DIR / f"{page_id}.mmd").read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(ok=False, payload={"reason": f"could not read page {page_id}: {exc}"})
        return ToolResult(ok=True, payload={"page_id": page_id, "text": text})


class EnhancePage(Tool):
    name = "enhance_page"

    def __call__(self, page_id: str) -> ToolResult:  # type: ignore[override]
        if not page_id:
            return ToolResult(ok=False, payload={"reason": "page_id must be non-empty"})
        # Honest no-op, not faked work: A1's own trade-off (configs/config.yaml
        # `enhance: {enabled: false}`) is that our 1964 scans are already clean, and a
        # generative repair model can hallucinate a stroke into a wrong formula -- so
        # classical cleanup is skipped by design. Read the flag live rather than hardcode
        # "always false", so this stays honest if that config value ever changes.
        enabled = bool(config.load().get("enhance", {}).get("enabled", False))
        if not enabled:
            return ToolResult(
                ok=True,
                payload={
                    "page_id": page_id,
                    "enhanced": False,
                    "reason": "enhance.enabled=false (A1 trade-off: clean scans, "
                    "no generative-repair risk)",
                },
            )
        # If a future config ever turns enhancement back on, this is the one place a real
        # classical-cleanup call would be wired in -- not built now, since a near-no-op *is*
        # the current honest behaviour, not a placeholder standing in for missing work.
        return ToolResult(
            ok=True,
            payload={
                "page_id": page_id,
                "enhanced": False,
                "reason": "enhance.enabled=true in config, but no cleanup implementation exists yet",
            },
        )


_FORMULA_ID = re.compile(r"^\d+\.\d+\.\d+$")


class Extract(Tool):
    name = "extract"

    def __call__(self, field: str, chunk_id: str) -> ToolResult:  # type: ignore[override]
        chunk = _lookup_chunk(chunk_id)
        if chunk is None:
            return ToolResult(ok=False, payload={"reason": f"no such chunk: {chunk_id!r}"})
        if _FORMULA_ID.match(field):
            # A&S numbered formula: reuse eval/metrics.py's own parser (three known layouts,
            # already tested) instead of re-solving the same problem here. `body` is
            # normalised (extract_formulas' own contract), so it is not a literal substring
            # of chunk.text -- the span still anchors to something real, though: where the
            # id itself appears in the raw text.
            body = extract_formulas(chunk.text).get(field)
            if body is not None:
                m = re.search(re.escape(field), chunk.text)
                span = (m.start(), m.end()) if m else None
                return ToolResult(
                    ok=True,
                    payload={"chunk_id": chunk_id, "field": field, "value": body, "span": span},
                )
        # Fallback: literal substring search -- covers table values, prose terms, and any
        # formula id that didn't parse under extract_formulas' three known layouts.
        start = chunk.text.find(field)
        if start == -1:
            return ToolResult(
                ok=False, payload={"reason": f"{field!r} not found in chunk {chunk_id}"}
            )
        end = start + len(field)
        return ToolResult(
            ok=True,
            payload={
                "chunk_id": chunk_id,
                "field": field,
                "value": chunk.text[start:end],
                "span": (start, end),
            },
        )


_NUMERIC_OPS: dict[str, Callable[[list[float]], float]] = {
    "sum": sum,
    "mean": lambda xs: sum(xs) / len(xs),
    "max": max,
    "min": min,
}


class Aggregate(Tool):
    name = "aggregate"

    def __call__(self, op: str, items: list) -> ToolResult:  # type: ignore[override]
        if op == "count":
            return ToolResult(ok=True, payload={"op": op, "value": len(items)})
        if op == "concat":
            return ToolResult(ok=True, payload={"op": op, "value": " ".join(str(i) for i in items)})
        if op not in {"sum", "mean", "max", "min"}:
            return ToolResult(ok=False, payload={"reason": f"unknown aggregate op: {op!r}"})
        if not items:
            return ToolResult(ok=False, payload={"reason": "no items to aggregate"})
        try:
            numeric = [float(x) for x in items]
        except (TypeError, ValueError) as exc:
            return ToolResult(
                ok=False, payload={"reason": f"non-numeric item for op {op!r}: {exc}"}
            )
        value = _NUMERIC_OPS[op](numeric)
        return ToolResult(ok=True, payload={"op": op, "value": value, "n": len(numeric)})


class Cite(Tool):
    name = "cite"

    def __call__(self, chunk_id: str, span: tuple) -> ToolResult:  # type: ignore[override]
        chunk = _lookup_chunk(chunk_id)
        if chunk is None:
            return ToolResult(ok=False, payload={"reason": f"no such chunk: {chunk_id!r}"})
        try:
            start, end = span
        except (TypeError, ValueError):
            return ToolResult(
                ok=False, payload={"reason": f"span must be a (start, end) pair, got {span!r}"}
            )
        if start < 0 or end <= start or end > len(chunk.text):
            return ToolResult(
                ok=False,
                payload={
                    "reason": f"span {span!r} is out of bounds for chunk {chunk_id} "
                    f"(len={len(chunk.text)})"
                },
            )
        citation = Citation(chunk_id=chunk_id, span=(start, end))
        pages = ", ".join(chunk.page_ids)
        # Lighter than SYNTHESIZE's own "why this reference over the runner-up" line (that
        # one is LLM-authored and can see the full candidate list) -- this tool only sees one
        # chunk_id, so its rationale is a deterministic, chunk-grounded justification, useful
        # when `cite` is invoked directly as a DECIDE-routed action outside the SYNTHESIZE flow.
        rationale = (
            f"Cited chunk {chunk_id} (page(s) {pages}, retrieval score {chunk.score:.3f}) "
            f"— this span is the text the claim is drawn from."
        )
        return ToolResult(
            ok=True, payload={"citation": citation.model_dump(), "rationale": rationale}
        )


# --- sandboxed calculator ------------------------------------------------------------------
#
# No eval()/exec() on model-generated text -- that is a real injection hole (a model could be
# steered into emitting `__import__('os').system(...)`), and bandit flags the lazy version
# (B307). ast.parse(..., mode="eval") + a hand-walked, explicitly whitelisted node evaluator
# is the standard safe pattern: only arithmetic, a small constant/function whitelist, nothing
# that can reach an attribute, a subscript, a name lookup outside the whitelist, or a call to
# anything not in it.


class _UnsafeExpressionError(Exception):
    pass


_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS: dict[type, Any] = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_CONSTANTS = {"pi": math.pi, "e": math.e}
# A1's own worked example: Gamma(1/2) ~= 1.77245 -- gamma/sqrt must both be reachable.
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "gamma": math.gamma,
    "factorial": math.factorial,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "degrees": math.degrees,
    "radians": math.radians,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise _UnsafeExpressionError(f"non-numeric literal: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
            raise _UnsafeExpressionError("only whitelisted function calls are allowed")
        args = [_safe_eval(a) for a in node.args]
        return _FUNCTIONS[node.func.id](*args)
    raise _UnsafeExpressionError(f"disallowed expression: {type(node).__name__}")


class Calculator(Tool):
    name = "calculator"

    def __call__(self, expr: str) -> ToolResult:  # type: ignore[override]
        try:
            tree = ast.parse(expr, mode="eval")
        except (SyntaxError, ValueError) as exc:
            return ToolResult(ok=False, payload={"reason": f"could not parse expression: {exc}"})
        try:
            value = _safe_eval(tree)
        except _UnsafeExpressionError as exc:
            return ToolResult(ok=False, payload={"reason": str(exc)})
        except ZeroDivisionError:
            return ToolResult(ok=False, payload={"reason": "division by zero"})
        except (ValueError, OverflowError, TypeError) as exc:
            return ToolResult(ok=False, payload={"reason": f"math error: {exc}"})
        return ToolResult(ok=True, payload={"expr": expr, "value": value})


class EscalateToHuman(Tool):  # HITL entry
    name = "escalate_to_human"

    def __call__(self, reason: str, context: dict) -> ToolResult:  # type: ignore[override]
        try:
            return hitl.escalate(reason, context)
        except Exception as exc:  # noqa: BLE001
            # hitl.escalate is Step 13's job and currently raises NotImplementedError -- this
            # tool's own "never raise" contract has to hold regardless of whether that step
            # has landed yet, so any exception from the delegate becomes ok=False here, not a
            # crash. Once Step 13 implements escalate() for real, this passes its result
            # through untouched.
            return ToolResult(ok=False, payload={"reason": f"escalation failed: {exc}"})


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
