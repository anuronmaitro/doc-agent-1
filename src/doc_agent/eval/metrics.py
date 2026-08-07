"""Stage 9 — metrics"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


def ocr_f1(pred: str, gold: str) -> float:
    raise NotImplementedError


def recall_at_k(retrieved: list, gold: list, k: int) -> float:
    raise NotImplementedError


def groundedness(answer: Answer) -> float:
    raise NotImplementedError  # no-hallucination


def citation_accuracy(answer: Answer) -> float:
    raise NotImplementedError


def ece(confidences: Any, correct: Any) -> float:
    raise NotImplementedError  # calibration


def subgroup_gap(scores_by_group: dict) -> float:
    raise NotImplementedError  # fairness
