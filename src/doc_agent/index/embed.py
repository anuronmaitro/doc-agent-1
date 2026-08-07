"""Stage 4 — embed chunks"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


def encode(chunks: list[Chunk], cfg: dict) -> Any:
    """Embed with cfg['embed']['model']. IMPLEMENT."""
    raise NotImplementedError("Stage 4: embed")
