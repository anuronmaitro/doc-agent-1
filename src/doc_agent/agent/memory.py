"""Stage 6 — working/episodic memory"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import *  # noqa

_WORD = re.compile(r"[a-zA-Z0-9]+")


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


class Memory:
    """Working memory for ONE query's agent loop -- a fresh `Memory()` per query, not a
    cross-query cache (plan_a3.md Step 8). `add()` is called once per loop step with
    whatever that step actually produced (a `ToolResult`, a dict, a plain string -- the
    loop decides the shape, this class doesn't require one), so `recall` treats items as
    opaque and scores them via `str(item)`.
    """

    def __init__(self) -> None:
        self.items: list = []

    def add(self, item: Any) -> None:
        self.items.append(item)

    def recall(self, query: str) -> list:
        """Prior observations from this run relevant to `query`, most-relevant-first.

        Deliberately simple, not clever (plan_a3.md Step 8 is explicit about this): Jaccard
        word overlap between `query` and each stored item's `str()` -- shared distinct words
        over the union of both. No semantics, no embeddings; just enough to prefer an
        observation that is actually about what's being asked over one that isn't. Items
        with zero shared words are dropped rather than returned in meaningless order, so an
        empty result cleanly means "nothing in memory is relevant" -- not "everything is,
        equally." Empty memory or an empty/blank query both return `[]`.
        """
        query_words = _words(query)
        if not self.items or not query_words:
            return []
        scored: list[tuple[float, int, Any]] = []
        for order, item in enumerate(self.items):
            item_words = _words(str(item))
            overlap = query_words & item_words
            if not overlap:
                continue
            score = len(overlap) / len(query_words | item_words)
            scored.append((score, order, item))
        # Ties broken by insertion order (earlier observation first) -- sort is stable on
        # score alone, but negating `order` this way makes the tie-break explicit rather
        # than an implementation detail of Python's sort to rely on silently.
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [item for _, _, item in scored]
