"""Stage 7 — RLVR — verifiable reward on extraction accuracy"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


def verifiable_reward(prediction: Any, gold: Any) -> float:
    """+1 if extraction exactly matches gold, else 0. Drives RLVR/GRPO. IMPLEMENT."""
    raise NotImplementedError("Stage 7: verifiable reward")
