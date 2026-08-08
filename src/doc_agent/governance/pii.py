"""Governance — PII detection + redaction (mandatory)"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# data/provenance.md documents exactly two real people in this corpus: the two
# editors credited on the title page. Everything else in the 1046 content pages is
# formulas/tables, where capitalized word-pairs are function names ("Bessel
# Function", "Legendre Polynomial", ...) rather than people — a generic
# capitalized-bigram name regex would redact those and corrupt citation text. So
# person detection stays deliberately narrow: this known-name list, plus a
# title-anchored pattern (Dr./Prof./Mr./Mrs./Miss/Editor ...) for any other named
# individual that might show up in scanned front matter.
_KNOWN_PERSON_NAMES = (
    "Milton Abramowitz",
    "Irene A. Stegun",
)

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

_TITLED_NAME_RE = re.compile(
    r"\b(?:Dr|Prof|Professor|Mr|Mrs|Ms|Miss|Editor)\.?\s+"
    r"[A-Z][a-zA-Z'-]+(?:\s+[A-Z]\.)*(?:\s+[A-Z][a-zA-Z'-]+)+"
)

_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-zA-Z]*\s){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b\.?"
)


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return non-overlapping (start, end, type) PII spans found in text."""
    spans: list[tuple[int, int, str]] = []
    for m in _EMAIL_RE.finditer(text):
        spans.append((m.start(), m.end(), "EMAIL"))
    for m in _ADDRESS_RE.finditer(text):
        spans.append((m.start(), m.end(), "ADDRESS"))
    for m in _TITLED_NAME_RE.finditer(text):
        spans.append((m.start(), m.end(), "PERSON"))
    for name in _KNOWN_PERSON_NAMES:
        start = 0
        idx = text.find(name, start)
        while idx != -1:
            spans.append((idx, idx + len(name), "PERSON"))
            start = idx + len(name)
            idx = text.find(name, start)
    return _merge_overlaps(sorted(spans))


def _merge_overlaps(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Spans arrive sorted by start; collapse overlapping/adjacent spans."""
    merged: list[tuple[int, int, str]] = []
    for start, end, kind in spans:
        if merged and start <= merged[-1][1]:
            prev_start, prev_end, prev_kind = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), prev_kind)
        else:
            merged.append((start, end, kind))
    return merged


def redact(text: str) -> str:
    """Replace every detected PII span with a [REDACTED:TYPE] marker."""
    spans = detect(text)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for start, end, kind in spans:
        out.append(text[cursor:start])
        out.append(f"[REDACTED:{kind}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def _redact_in_place(value: Any) -> None:
    """Walk a hook ctx and scrub every string it finds, mutating containers in
    place. Mutation (not rebuild-and-return) matters: pipeline.build_knowledge_base
    and Agent.run both discard hooks.run()'s return value and keep using their own
    local `text` / `state` variables afterwards, and those are the *same* list/dict
    objects handed to us as ctx — so only in-place edits actually reach them."""
    if isinstance(value, Chunk):
        value.text = redact(value.text)
    elif isinstance(value, ToolResult):
        _redact_in_place(value.payload)
    elif isinstance(value, dict):
        for k, v in value.items():
            if isinstance(v, str):
                value[k] = redact(v)
            else:
                _redact_in_place(v)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            if isinstance(v, str):
                value[i] = redact(v)
            else:
                _redact_in_place(v)
    # numbers/bool/None/unrecognized objects are left alone on purpose: redacting
    # fields we don't understand risks corrupting non-text data instead of PII.


def register(hooks: Any) -> None:
    """Wire PII redaction into the pipeline at every seam that can see corpus or
    conversation text before it leaves the system."""

    def _scrub(ctx: dict) -> dict:
        try:
            _redact_in_place(ctx)
        except Exception:
            logger.warning("pii._scrub: failed to redact ctx; passing it through unscrubbed")
        return ctx

    hooks.register(hooks.AFTER_OCR, _scrub)  # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)  # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)  # scrub logs
