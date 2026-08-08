"""Stage 4 — chunk text (semantic chunking, bonus E4)"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# A&S numbers formulas chapter.section.formula (6.1.8); that id is our citation anchor
# (summary.md 3f). vision/ocr.py and eval/metrics.py each already match this exact
# pattern at line-start -- reused here (not imported: eval's copy is module-private, and
# each pipeline stage staying self-contained is the point of the 9-stage structure) so
# all three stages agree on what counts as a formula line.
FORMULA_LINE_RE = re.compile(r"^\s*(\d+\.\d+\.\d+)\s+(\S.*)$")

# Nougat renders section titles as markdown headings (the same pattern eval.metrics
# strips before scoring) -- that markup is our prose-split signal.
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")

# cfg["index"]["chunk_tokens"] (config.yaml, locked at 512) is a *ceiling*, not the split
# strategy -- see split()'s docstring for why overlap stays 0 even so.
DEFAULT_MAX_TOKENS = 512

# A source chunk (one per Step 10 region) whose id already carries a guessed formula id
# (vision.ocr._chunk_id: the *first* id found anywhere in the whole region) has this
# shape; this stage recomputes one id per formula it actually finds, replacing the guess.
_ID_FORMULA_SUFFIX_RE = re.compile(r"^(.*)\|(\d+\.\d+\.\d+)$")


@dataclass
class _Segment:
    """One structural piece of a chunk's text: either a numbered formula (its defining
    line plus any immediately-indented continuation, e.g. its condition of validity) or
    a run of prose bounded by the next heading/formula."""

    lines: list[str] = field(default_factory=list)
    formula_id: str | None = None


def _segment_text(text: str) -> list[_Segment]:
    """Split raw OCR text into formula and prose segments.

    A formula segment opens at a line starting with its id (FORMULA_LINE_RE) and keeps
    only *indented* continuation lines -- typically the condition of validity printed
    directly under the formula, e.g. "(Rz>0)" -- closing at a blank line or the first
    unindented line, exactly mirroring eval.metrics.extract_formulas' own closing rule
    (the two must agree on what a formula block is, since one writes gold labels and the
    other parses model output for the same alignment key).

    A prose segment opens at start-of-text, right after a formula segment closes, or at
    a heading line -- and then just keeps accumulating, blank lines included, until the
    next heading or formula id. Splitting happens *at* headers, not at every paragraph
    break, per plan.md Step 13.
    """
    segments: list[_Segment] = []
    current: _Segment | None = None

    def open_segment(formula_id: str | None, first_line: str) -> _Segment:
        seg = _Segment(lines=[first_line] if first_line.strip() else [], formula_id=formula_id)
        segments.append(seg)
        return seg

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        m = FORMULA_LINE_RE.match(line)
        if m:
            current = open_segment(m.group(1), line)
            continue
        if HEADING_RE.match(line):
            current = open_segment(None, line)
            continue
        if current is not None and current.formula_id is not None:
            if line.strip() and line[:1].isspace():
                current.lines.append(line)  # indented -> condition of validity, stays
                continue
            if not line.strip():
                current = None  # blank line closes the formula
                continue
            current = open_segment(None, line)  # unindented -> new prose starts here
            continue
        if current is None:
            current = open_segment(None, line)
        else:
            current.lines.append(line)

    return [s for s in segments if any(ln.strip() for ln in s.lines)]


def _size_guard(text: str, max_tokens: int) -> list[str]:
    """A segment that already fits is returned as-is. Otherwise fall back to bounded
    pieces so one pathological block (an unbroken wall of prose with no headings, or a
    giant table dumped as a single region) can never produce an unbounded chunk. This is
    a safety net, not the primary strategy: real formula blocks and header-delimited
    prose are already naturally bounded on A&S pages.

    Packs whole lines up to the ceiling, but a single line that alone exceeds it (a
    realistic case: Nougat often emits one whole paragraph as one unbroken line, so
    line-boundaries alone are not a reliable bound) falls back to a word-level split."""
    if len(text.split()) <= max_tokens:
        return [text]
    pieces: list[str] = []
    cur: list[str] = []
    cur_words = 0

    def flush() -> None:
        nonlocal cur, cur_words
        if cur:
            pieces.append("\n".join(cur))
            cur, cur_words = [], 0

    for line in text.splitlines():
        words = line.split()
        if len(words) > max_tokens:
            flush()
            for i in range(0, len(words), max_tokens):
                pieces.append(" ".join(words[i : i + max_tokens]))
            continue
        if cur and cur_words + len(words) > max_tokens:
            flush()
        cur.append(line)
        cur_words += len(words)
    flush()
    return pieces


def _base_id(chunk_id: str) -> str:
    m = _ID_FORMULA_SUFFIX_RE.match(chunk_id)
    return m.group(1) if m else chunk_id


def _split_one(chunk: Chunk, max_tokens: int) -> list[Chunk]:
    """Re-chunk one Step-11 region chunk onto its own structure. `doc_id` and `page_ids`
    are carried through unchanged (plan.md); `id` is preserved exactly when nothing
    actually needed splitting, and otherwise extends the source id with the formula id
    (`...|6.1.8`) or a part index (`...|p00`) it produced."""
    pieces: list[tuple[str, str | None]] = []
    for seg in _segment_text(chunk.text):
        seg_text = "\n".join(seg.lines).strip()
        if not seg_text:
            continue
        for part in _size_guard(seg_text, max_tokens):
            pieces.append((part, seg.formula_id))

    if not pieces:
        return []

    base = _base_id(chunk.id)
    if len(pieces) == 1:
        text, formula_id = pieces[0]
        new_id = f"{base}|{formula_id}" if formula_id else base
        return [
            Chunk(
                id=new_id,
                doc_id=chunk.doc_id,
                text=text,
                page_ids=chunk.page_ids,
                score=chunk.score,
            )
        ]

    out: list[Chunk] = []
    seen: dict[str, int] = {}
    part_n = 0
    for text, formula_id in pieces:
        new_id = f"{base}|{formula_id}" if formula_id else f"{base}|p{part_n:02d}"
        if formula_id is None:
            part_n += 1
        if new_id in seen:  # a repeated formula id on one source chunk is rare but not
            seen[new_id] += 1  # impossible (a cross-reference restatement) -- keep ids unique
            new_id = f"{new_id}-{seen[new_id]}"
        else:
            seen[new_id] = 0
        out.append(
            Chunk(
                id=new_id,
                doc_id=chunk.doc_id,
                text=text,
                page_ids=chunk.page_ids,
                score=chunk.score,
            )
        )
    return out


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Semantic chunking (bonus E4): re-chunk Step 11's per-region OCR chunks onto the
    document's own structure, not fixed windows -- one chunk per numbered formula block
    plus its immediate conditions of validity, prose split at section headers, and a
    max-size guard so a pathological block still stays bounded (`_size_guard`).
    `cfg["index"]["chunk_tokens"]` is that ceiling, never the primary split point, which
    is why config.yaml locks `overlap: 0` (summary.md 7a): a structural split doesn't
    need overlap to avoid cutting a formula in half, only a sliding window would.
    """
    max_tokens = int((cfg.get("index") or {}).get("chunk_tokens", DEFAULT_MAX_TOKENS))
    out: list[Chunk] = []
    for c in chunks:
        out.extend(_split_one(c, max_tokens))
    return out
