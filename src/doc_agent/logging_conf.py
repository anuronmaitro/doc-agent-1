"""FIXED — structured logging (auditable NFR). Use get_logger(), never print()."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from .contracts import TraceStep

# `__file__`-anchored, not cwd-relative: a Kaggle run's cwd is /kaggle/working/repo while a
# local dev/test run's is wherever pytest/python was launched from -- neither is reliable,
# but the repo's own on-disk layout (this file always sits at src/doc_agent/logging_conf.py
# relative to the repo root) is, on both.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACE_PATH = REPO_ROOT / "traces" / "run.jsonl"


def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter(
                '{"ts":"%(asctime)s","lvl":"%(levelname)s","mod":"%(name)s","msg":"%(message)s"}'
            )
        )
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg


logger = get_logger(__name__)


def register(hooks: Any) -> None:
    """Wire structured tracing at ON_STEP / ON_TOOL_CALL / AFTER_ANSWER, emitting
    traces/run.jsonl (schema: traces/README.md, matches contracts.TraceStep) so the A3
    agentic-feature gate can read the real trajectory.

    Three different ctx shapes arrive here, and `_trace` must never raise for any of them --
    a trace handler that throws takes the whole agent down with it (the same "never raise at
    a hook seam" contract pii.py's `_scrub` and postprocess.py's `_ground` already hold to):
      - ON_STEP, `{"state": state}` -- fires once at the top of each `Agent.run()` call,
        BEFORE `decide()` has done anything. Nothing to write yet; this is purely where a
        NEW run is recognised (a new `state` object) so the trace file starts fresh for it.
      - ON_TOOL_CALL, `{"action": action}` -- fires once per REGISTRY-routed tool call
        (before `act()` runs it). Always written as its own step.
      - AFTER_ANSWER, `{"answer": answer}` -- fires once, after `synthesize()`. By now
        `decide()`'s own internal evidence-gated re-search (Step 10) has fully populated
        `state["obs"]` with one `{"top_score", "k"}` entry per retrieval attempt -- flushed
        here as one `retrieve` TraceStep each (ON_STEP fired too early to see any of this),
        followed by one final `answer` TraceStep.

    **Truncate-then-append, per run, not per process:** the file is truncated the moment a
    NEW `state` object is seen at ON_STEP, then appended to for the rest of that same run.
    Deliberate, not accidental (plan_a3.md Step 12 Do item 6) -- Step 25 later commits
    `traces/run.jsonl` as evidence of a real run, and that commit needs to be exactly the
    most recent run's trail, not an accumulation left over from earlier dev/test iterations.
    """
    tracker: dict[str, Any] = {"state": None, "step": 0, "started": False}

    def _next_step() -> int:
        tracker["step"] += 1
        return tracker["step"]

    def _append(tool: str, args: dict, obs: dict) -> None:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if tracker["started"] else "w"
        tracker["started"] = True
        entry = TraceStep(step=_next_step(), tool=tool, args=args, obs=obs)
        with TRACE_PATH.open(mode, encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")

    def _trace(ctx: dict) -> dict:
        try:
            if "state" in ctx:
                state = ctx["state"]
                if state is not tracker["state"]:
                    tracker["state"] = state
                    tracker["step"] = 0
                    tracker["started"] = False
            elif "action" in ctx:
                action = ctx["action"]
                _append(action.get("tool", "?"), action.get("args", {}), {})
            elif "answer" in ctx:
                state = tracker["state"] or {}
                query = state.get("query", "")
                for obs in state.get("obs", []):
                    if isinstance(obs, dict) and "top_score" in obs and "k" in obs:
                        _append("retrieve", {"query": query, "k": obs["k"]}, obs)
                answer = ctx["answer"]
                _append("answer", {}, {"abstained": not answer.grounded})
        except Exception:
            logger.warning("logging_conf._trace: failed to append a trace step; skipping it")
        return ctx

    hooks.register(hooks.ON_STEP, _trace)
    hooks.register(hooks.ON_TOOL_CALL, _trace)
    hooks.register(hooks.AFTER_ANSWER, _trace)
