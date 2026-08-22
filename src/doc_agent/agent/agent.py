"""Stage 6 - FIXED loop - perceive -> decide -> act -> observe, with cross-cutting seams.
Implement synthesize() only (decide() and act() are both done). Security, grounding, PII, and
tracing run via hooks at the marked seams - do NOT inline them here."""

from __future__ import annotations

from typing import Any

from .. import hooks
from ..contracts import *  # noqa
from ..retrieval import retriever as retriever_mod
from . import tools
from .memory import Memory


class Agent:
    """FIXED loop. Implement synthesize() (the grounded answer) only."""

    def __init__(self, cfg: dict, retriever: Any) -> None:
        self.cfg = cfg["agent"]
        # decide()'s evidence-gated re-search needs cfg["retrieve"] (k/k_step/k_max/
        # weak_threshold), which self.cfg above doesn't carry -- same "keep the whole cfg
        # around" pattern retrieval/retriever.py's own Retriever.__init__ already uses
        # (its self._full_cfg) for the identical reason.
        self._full_cfg = cfg
        self.retriever = retriever
        self.mem = Memory()

    def run(self, query_text: str) -> Answer:
        state: dict[str, Any] = {"query": query_text, "obs": []}
        for _ in range(self.cfg["max_steps"]):
            hooks.run(hooks.ON_STEP, {"state": state})
            action = self.decide(state)  # IMPLEMENT (policy)
            if action["tool"] == "stop":
                break
            hooks.run(hooks.ON_TOOL_CALL, {"action": action})  # guardrails/injection/trace
            result = self.act(action)  # runs the tool via REGISTRY
            state["obs"].append(result)
            self.mem.add(result)
        hooks.run(hooks.BEFORE_ANSWER, {"state": state})  # grounding gate / PII redact
        ans = self.synthesize(state)  # IMPLEMENT (grounded answer)
        hooks.run(hooks.AFTER_ANSWER, {"answer": ans})  # trace / metrics
        return ans

    def decide(self, state: dict) -> dict:
        """Evidence-gated re-search — the MANDATORY agentic behaviour (A3 gate, fail-closed).
        Read the last observation (top_score, k) and branch on the NUMBER, using retrieval.retriever:
          1. retrieve at k = cfg.retrieve.k
          2. if is_weak(chunks, cfg):  k2 = next_k(k, cfg)
               - k2 is not None -> retrieve AGAIN at the wider k2 (widen the net), then re-check
               - k2 is None (hit k_max) and still weak -> ABSTAIN ("insufficient evidence")
          3. else -> synthesize a grounded, cited answer
        Emit obs {"top_score": ..., "k": ...} on each step. A fixed retrieve->answer path is NOT agentic
        and caps the grade. Rule-based (baseline) or RL policy (Stage 7)."""
        rcfg = self._full_cfg
        query = state["query"]

        def _emit(chunks: list, k: int) -> None:
            state["obs"].append({"top_score": retriever_mod.top_score(chunks), "k": k})

        k = rcfg["retrieve"]["k"]
        chunks = self.retriever.retrieve(query, k=k)
        while retriever_mod.is_weak(chunks, rcfg):
            _emit(chunks, k)
            k2 = retriever_mod.next_k(k, rcfg)
            if k2 is None:
                # Hit k_max still weak -> ABSTAIN. state["chunks"] keeps the last (still
                # weak) attempt so synthesize() can show its work / cite why it abstained,
                # not because those chunks are meant to ground an answer.
                state["chunks"] = chunks
                state["abstain"] = True
                state["abstain_reason"] = "insufficient evidence"
                return {"tool": "stop", "args": {}}
            k = k2
            chunks = self.retriever.retrieve(query, k=k)
        _emit(chunks, k)
        state["chunks"] = chunks
        state["abstain"] = False
        return {"tool": "stop", "args": {}}

    def act(self, action: dict) -> ToolResult:
        """Look `action["tool"]` up in `tools.REGISTRY` by name and call it with
        `action["args"]`. An unknown tool name returns `ToolResult(ok=False, ...)`, same
        "never raise mid-run" contract every tool in the registry already honours (Step 7) --
        a bad dispatch must not crash the loop any more than a bad tool call would."""
        name = action["tool"]
        for tool_cls in tools.REGISTRY:
            if tool_cls.name == name:
                # REGISTRY's element type widens to type[Tool] (the abstract base) once
                # mixed concrete subclasses join in a list literal -- every entry is
                # concrete in practice (test_tools.py's own test_registry_is_tool_subclasses
                # asserts issubclass), mypy just can't see that through the join.
                return tool_cls()(**action.get("args", {}))  # type: ignore[abstract]
        return ToolResult(ok=False, payload={"reason": f"unknown tool: {name!r}"})

    def synthesize(self, state: dict) -> Answer:
        """Grounded, cited answer; abstain if unsupported (no-hallucination)."""
        raise NotImplementedError("Stage 6: synthesize grounded answer")
