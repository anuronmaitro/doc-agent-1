"""MLOps — experiment tracking (W&B)"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


def init_run(cfg: dict, tags: list[str]) -> Any:
    """Start a W&B run; log config. IMPLEMENT."""
    raise NotImplementedError("MLOps: init_run")


def log(metrics: dict) -> None:
    raise NotImplementedError("MLOps: log")
