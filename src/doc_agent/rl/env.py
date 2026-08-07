"""Stage 7 — Gymnasium env for tool/retrieval selection"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from ..contracts import *  # noqa


class ToolSelectionEnv(gym.Env):
    """State=agent context, Action=tool choice, Reward=task success under budget. IMPLEMENT."""

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Any:
        raise NotImplementedError("Stage 7: env.reset")

    def step(self, action: Any) -> Any:
        raise NotImplementedError("Stage 7: env.step")
