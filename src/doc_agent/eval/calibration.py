"""Stage 9 — confidence calibration (calibrated-confidence NFR)"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


def temperature_scale(logits: Any, labels: Any) -> Any:
    """Fit temperature on val; return scaler. IMPLEMENT."""
    raise NotImplementedError("Calibration: temperature scaling")


def ece(confidences: Any, correct: Any) -> float:
    raise NotImplementedError("Calibration: ECE")
