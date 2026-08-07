"""Stage 8 — affordable adaptation — LoRA / quantization"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


def apply_lora(model: Any, cfg: dict) -> Any:
    """Wrap a component with LoRA per cfg. IMPLEMENT."""
    raise NotImplementedError("Adapt: LoRA")


def quantize(model: Any, cfg: dict) -> Any:
    """Post-training quantization per cfg. IMPLEMENT."""
    raise NotImplementedError("Adapt: quantize")
