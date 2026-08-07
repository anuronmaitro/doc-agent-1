"""Stage 4 — vector store"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa


def build(chunks: list[Chunk], vectors: Any, cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type']). IMPLEMENT."""
    raise NotImplementedError("Stage 4: build index")


def load(cfg: dict) -> Any:
    raise NotImplementedError("Stage 4: load index")
