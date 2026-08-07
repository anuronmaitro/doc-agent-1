"""OPTIONAL — batch/streaming ingestion for very large corpora
Activate only if your data speciality or NFR requires it (e.g. a huge-corpus / scalable project). Off by default; CI does not require impl.
"""

from __future__ import annotations

from typing import Any


def stream(cfg: dict) -> Any:
    raise NotImplementedError("optional: stream ingest")
