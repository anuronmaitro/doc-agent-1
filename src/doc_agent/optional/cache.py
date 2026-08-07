"""OPTIONAL — embedding/retrieval/answer cache
Activate only if your data speciality or NFR requires it (e.g. a low-latency NFR). Off by default; CI does not require impl.
"""

from __future__ import annotations

from typing import Any


def get(key: str) -> Any:
    raise NotImplementedError("optional: cache get")


def put(key: str, val: Any) -> None:
    raise NotImplementedError("optional: cache put")
