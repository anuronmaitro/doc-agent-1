"""HITL — human-in-the-loop review queue"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from . import hitl_store

logger = get_logger(__name__)


def escalate(reason: str, context: dict) -> ToolResult:
    """Queue for human review; block action until approved.

    "Blocks" here means the disputed action does not proceed on its own say-so -- the CALLER
    (guardrails.Guardrails.check() raising before act() runs it, or synthesize() returning an
    abstention instead of a low-confidence answer) is what actually stops it; this function's
    own job is narrower and more mechanical: reliably get the item queued. Never raises --
    same "a cross-cutting handler must not take the agent down" contract every other hook
    handler in this codebase already holds to (governance/pii.py's _scrub, llm/postprocess.py's
    _ground) -- an escalation failing to queue is a real problem worth logging loudly, but it
    must not additionally crash the run that triggered it.
    """
    try:
        item_id = hitl_store.enqueue({"reason": reason, "context": context})
    except Exception as exc:  # noqa: BLE001 -- see docstring: this must never raise
        logger.error(f"hitl.escalate: failed to queue for review: {exc}")
        return ToolResult(ok=False, payload={"reason": f"escalation failed: {exc}"})
    logger.warning(f"hitl.escalate: queued item {item_id} for human review: {reason}")
    return ToolResult(ok=True, payload={"escalated": True, "item_id": item_id, "reason": reason})


def review_queue() -> Any:
    """Return pending items for the reviewer UI."""
    return hitl_store.pending()
