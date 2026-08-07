"""Stage 7 — policy network"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


class Policy:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["rl"]

    def act(self, state: Any) -> dict:
        raise NotImplementedError("Stage 7: policy.act")
