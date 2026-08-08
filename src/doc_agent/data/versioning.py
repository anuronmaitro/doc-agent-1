"""Data — corpus versioning (which corpus version -> which result)"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..logging_conf import get_logger

logger = get_logger(__name__)


def snapshot(corpus_dir: str, *, out_path: str | Path = "data/corpus_version.json") -> str:
    """Hash a corpus directory -> a short, reproducible version id.

    Hashes (relative_path, size_bytes) for every file under corpus_dir — not file
    contents: the rendered corpus is ~1 GB, so content-hashing would make this too slow
    to rerun casually. Filenames + sizes are exactly what scripts/get_data.sh's own
    integrity check already relies on to catch a truncated/incomplete download, so this
    stays consistent with how we detect corpus corruption elsewhere.

    Returns a 12-hex-char id: short enough to quote in CHANGELOG.md, long enough that an
    accidental collision between two genuinely different corpora is not a practical
    concern. Also writes a small JSON record to `out_path` (default
    `data/corpus_version.json`, gitignored like every other derived data/ artifact) so a
    teammate can see *which* corpus produced a given result without re-hashing it.

    Raises FileNotFoundError / ValueError if corpus_dir is missing or empty.
    """
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"snapshot: corpus_dir does not exist: {root}")

    entries = sorted(
        (str(f.relative_to(root)).replace("\\", "/"), f.stat().st_size)
        for f in root.rglob("*")
        if f.is_file()
    )
    if not entries:
        raise ValueError(f"snapshot: no files found under {root}")

    h = hashlib.sha256()
    for rel_path, size in entries:
        h.update(f"{rel_path}\t{size}\n".encode())
    full_hash = h.hexdigest()
    version_id = full_hash[:12]

    total_bytes = sum(size for _, size in entries)
    record = {
        "version_id": version_id,
        "sha256_full": full_hash,
        "corpus_dir": str(root),
        "n_files": len(entries),
        "total_bytes": total_bytes,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    logger.info(
        f"snapshot: {version_id} ({len(entries)} files, {total_bytes / 1e6:.1f} MB) -> {out}"
    )
    return version_id
