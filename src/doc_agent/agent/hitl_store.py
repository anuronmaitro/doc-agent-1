"""HITL — persistent review queue (survives restarts)"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from ..contracts import *  # noqa

# __file__-anchored (this file is src/doc_agent/agent/hitl_store.py, four levels under the
# repo root), not cwd-relative -- same reasoning as logging_conf.py's TRACE_PATH: a Kaggle
# run's cwd differs from a local one, but the repo's own on-disk layout doesn't.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
QUEUE_PATH = REPO_ROOT / "data" / "hitl_queue.json"


def _load() -> list[dict]:
    if not QUEUE_PATH.exists():
        return []
    try:
        return json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # A corrupt or unreadable queue file must not crash whatever called into HITL --
        # treat it as an empty queue rather than propagate; the next enqueue() rewrites it.
        return []


def _save(items: list[dict]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(items, indent=2), encoding="utf-8")


def enqueue(item: dict) -> str:
    """Persist a pending review item; return its id. A plain JSON array on disk (Step 13's
    own docstring says "sqlite/json" -- json is the simpler of the two and this queue is
    small), read-modify-written whole rather than appended-to, since resolve() (not this
    step's scope) needs to find and update one entry by id later."""
    item_id = uuid.uuid4().hex[:12]
    items = _load()
    items.append({**item, "id": item_id, "status": "pending"})
    _save(items)
    return item_id


def pending() -> list[dict]:
    """Items still awaiting a human decision -- everything with status == "pending"."""
    return [item for item in _load() if item.get("status") == "pending"]


def resolve(item_id: str, decision: str) -> None:
    """Mark a queued item resolved. Not this step's scope (plan_a3.md Step 13's Do list
    covers enqueue/pending only, for check()/escalate() to use) -- left raising so a caller
    can't silently depend on a resolution path that doesn't exist yet."""
    raise NotImplementedError("HITL store: resolve")
