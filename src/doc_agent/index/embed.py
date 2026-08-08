"""Stage 4 — embed chunks"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# Embeddings are cached here, keyed by Chunk.id, so a re-run only encodes chunks that are
# new or whose text changed -- the same "resumable, don't pay twice" pattern vision/ocr.py
# uses for its .mmd cache. data/index/ is already gitignored (index/store.py, Step 15,
# rebuilds this stage's output from scratch too); a distinct filename avoids colliding
# with whatever store.py writes there.
CACHE_PATH = Path("data/index/embed_cache.npz")

DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIM = 1024
BATCH_SIZE = 32


def _seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    """Ids are stored as a plain unicode array (not dtype=object), so this never needs
    `allow_pickle=True` -- the cache is trusted-but-still-avoid-pickle-if-we-can."""
    if not path.exists():
        return {}
    try:
        data = np.load(path)
        ids = data["ids"].tolist()
        vectors = data["vectors"]
    except (OSError, KeyError, ValueError):
        logger.warning(f"index.embed: {path} is unreadable; re-encoding from scratch")
        return {}
    return {cid: vectors[i] for i, cid in enumerate(ids)}


def _save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.array(list(cache.keys()))
    vectors = np.stack(list(cache.values())).astype(np.float32)
    np.savez_compressed(path, ids=ids, vectors=vectors)


def encode(chunks: list[Chunk], cfg: dict) -> Any:
    """Embed with cfg['embed']['model'] (BAAI/bge-m3), batched, seeded, and
    L2-normalised for cosine -- index/store.py (Step 15) builds an IndexFlatIP on these,
    and inner product on unit vectors IS cosine similarity (summary.md 7a).

    Returns a float32 array of shape (len(chunks), cfg['embed']['dim']), aligned 1:1
    with `chunks` by position. Embeddings are cached to disk keyed by Chunk.id
    (CACHE_PATH), so re-running on a corpus where most chunks are unchanged only
    computes the chunks that are new or whose text changed since the last run.
    """
    embed_cfg = cfg.get("embed") or {}
    dim = int(embed_cfg.get("dim", DEFAULT_DIM))

    if not chunks:
        return np.zeros((0, dim), dtype=np.float32)

    _seed_all(int(cfg.get("seed", 42)))
    model_name = embed_cfg.get("model", DEFAULT_MODEL)

    cache = _load_cache(CACHE_PATH)
    missing = [c for c in chunks if cache.get(c.id) is None or cache[c.id].shape[0] != dim]

    if missing:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        texts = [c.text for c in missing]
        new_vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,  # unit vectors -> inner product == cosine
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        for c, v in zip(missing, new_vectors, strict=True):
            cache[c.id] = v
        _save_cache(CACHE_PATH, cache)
        logger.info(f"index.embed: encoded {len(missing)} new/changed chunks ({model_name})")
    else:
        logger.info("index.embed: all chunks already cached, nothing to encode")

    vectors = np.stack([cache[c.id] for c in chunks]).astype(np.float32)
    logger.info(f"index.embed: {vectors.shape[0]} chunks -> vectors {vectors.shape}")
    return vectors
