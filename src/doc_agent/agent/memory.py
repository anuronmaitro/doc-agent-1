"""Stage 6 — working/episodic memory"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


class Memory:
    def __init__(self) -> None:
        self.items: list = []

    def add(self, item: Any) -> None:
        self.items.append(item)

    def recall(self, query: str) -> list:
        raise NotImplementedError("Stage 6: memory recall")
