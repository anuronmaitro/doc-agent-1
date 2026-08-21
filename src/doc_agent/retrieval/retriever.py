"""Stage 5 — dense retrieval"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..index import store


class Retriever:
    """Loads the A2 index and the BGE-M3 encoder once, lazily, and reuses both across
    calls -- decide()'s evidence-gated re-search loop (A3 Step 10) calls retrieve()
    multiple times per query, so reloading a 16 MB index and a 568M-param encoder on
    every call would make the eval runs (Steps 22-24) unaffordable."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["retrieve"]
        self._full_cfg = cfg
        self._loaded: store.LoadedIndex | None = None
        self._encoder: Any | None = None

    def _ensure_loaded(self) -> store.LoadedIndex:
        if self._loaded is None:
            self._loaded = store.load(self._full_cfg)
        return self._loaded

    def _ensure_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            model_name = self._full_cfg["embed"]["model"]
            self._encoder = SentenceTransformer(model_name)
        return self._encoder

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Top-k dense retrieval. Set chunk.score (relevance) on every result so decide() can judge
        whether the evidence is weak.

        Embeds the query with the SAME model the index was built with (cfg['embed']['model'],
        BGE-M3) and L2-normalises it, exactly like index/embed.py does for chunks -- a mismatched
        model or a raw (un-normalised) query would silently turn cosine into a meaningless score
        (store.py's own _NORM_TOL comment explains why the index side already guards this).
        """
        k = k if k is not None else self.cfg["k"]
        loaded = self._ensure_loaded()
        encoder = self._ensure_encoder()

        qvec = encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, ids = loaded.index.search(qvec, k)

        results: list[Chunk] = []
        for score, i in zip(scores[0], ids[0], strict=True):
            if i < 0:
                continue  # FAISS pads with -1 when k exceeds ntotal
            # A fresh copy per result, never mutate loaded.chunks in place -- that list is
            # cached on the instance and shared across every call this Retriever makes, so
            # writing .score onto it directly would let one query's scores leak into another's.
            results.append(loaded.chunks[i].model_copy(update={"score": float(score)}))
        return results


# --- evidence-strength policy: read by agent.decide() for evidence-gated re-search ---
def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)


def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg.retrieve.weak_threshold."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]


def next_k(k: int, cfg: dict) -> int | None:
    """Widen the net: k + k_step, or None once it would exceed k_max (signal to ABSTAIN)."""
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None
