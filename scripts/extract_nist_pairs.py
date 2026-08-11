"""Step 25 — NIST Handbook extraction (Stage A pairs).

Renders (image, text) pairs from the NIST Handbook of Mathematical Functions (2010) PDF
for OCR fine-tuning "Stage A" — see plan.md Step 25 and the Steps 25-28 handoff-risk note.

IMPORTANT — measured deviation from plan.md's stated premise, found while building this:
plan.md describes the source as "LaTeX-native, so labels are exact by construction" and
says to "pull the matching text straight from the PDF content stream." That undersells what
the content stream actually contains. pdfTeX (this PDF's producer, confirmed via its own
metadata: "LaTeX with hyperref package" / "pdfTeX-1.40.3") does NOT embed LaTeX source or
macro structure in the PDF — only absolute-positioned glyphs, grouped into per-baseline
"lines" and, within those, "spans" that change font/size at each glyph run. There is no
"\\frac" or "^" anywhere in the content stream; that structure has to be RECONSTRUCTED from
position and size, or it is lost. Two reconstructions are implemented, both confirmed
against real rendered crops before being trusted:

- **Superscript/subscript** (`_markup_spans`): a span whose font size is well below the
  line's dominant size, offset above or below the dominant baseline, is a superscript or
  subscript -- e.g. the "s" in "f_s(z)" is CMMI7 (6.97pt) sitting ~1.5pt below the CMMI10
  (9.96pt) baseline of the rest of the line. Without this, `x^n` flattens to the plain,
  structurally-wrong string "xn".
- **Fractions** (`_try_fraction` / `_fraction_bars`): pdfTeX draws the vinculum (the bar
  in a\\frac) as an actual thin vector stroke, independent of the glyphs above and below
  it. `page.get_drawings()` finds it directly — a confirmed real fraction, not a guess from
  glyph geometry alone — so `\\frac{num}{den}` is only emitted when a matching bar is
  physically present between the two lines, at the right x-position. A formula that
  continues past the fraction on the same baseline as the denominator (e.g.
  "dw/dz + a^2w^2 = 1,") is split by comparing each glyph's x-position to the bar's
  x-range, not by any part of the line's own text.

What is still NOT reconstructed, on purpose: stacked sums/integrals/products (the "big
operator" itself is drawn with the unreliable CMEX font — see UNRELIABLE_FONT_PREFIXES —
and its limits sit at arbitrary offsets with no equivalent of a vinculum to confirm the
pairing) and radicals (a CMEX radical sign's extent is not delimited by anything in the
content stream — nothing marks how far under the "√" the expression goes). Guessing either
would reproduce exactly the silent-wrong-label failure this script was rewritten to avoid
(see data/README.md "Step 25" for the concrete pages that motivated the CMEX exclusion and
the fraction-bar check in the first place). Left as a documented gap, not attempted here.

A numbered equation is only paired when its entire body is reconstructable this way,
confirmed by real geometry -- multi-row constructs beyond a single confirmed fraction are
skipped rather than scrambled. See _extract_page_pairs for the exact accept/reject rules.

Usage:
    python scripts/extract_nist_pairs.py --cfg configs/nist_extract.yaml
    python scripts/extract_nist_pairs.py --cfg configs/nist_extract.yaml --limit-pages 50
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import yaml

# TeX math-font families (Computer Modern Math Italic/Symbol + AMS blackboard bold). A
# line containing ANY of these is math content, never prose, regardless of length or
# left-margin position -- found necessary after a margin+length-only prose heuristic
# misfired on a flush-left, un-indented display formula (see data/README.md "Step 25").
MATH_FONT_PREFIXES = ("CMMI", "CMSY", "MSBM")
# CMEX ("extension": radicals, big stretchy delimiters, large operators built from
# pieces) is excluded from pairs entirely, not just treated as non-prose -- pdfTeX's
# ToUnicode CMap for CMEX glyphs is unreliable (a tall radical sign was observed to
# decode as the literal letter "p"), so any formula touching it gets a silently wrong
# label rather than an obviously-missing one. Silently-wrong is the dangerous case for
# fine-tune labels, so this is a hard exclusion, not a skip-if-unsure.
UNRELIABLE_FONT_PREFIXES = ("CMEX",)

# Every numbered display in this handbook states a relation -- an equation, inequality,
# limit, or membership, never a bare expression. Requiring one of these symbols to appear
# in the final body_text is a cheap, well-supported plausibility check that catches a
# formula whose real content fell OUTSIDE walk_gap_pt in every direction, leaving only an
# isolated fragment near the id (e.g. a lone "0" subscript off an integral sign, or just
# a fraction's denominator with none of its numerator or "lhs =" prefix) that would
# otherwise look like a complete, if terse, plain single-line formula. Confirmed against
# NIST p.115 (3.11.26, truncated to a bare "0") and p.134/p.142 (4.21.40 / 4.35.37, each
# truncated to a lone denominator) -- both silently wrong until this check was added.
RELATIONAL_RE = re.compile(r"[=<>≤≥→↔∈∋≡∼≈∝⊂⊃⊆⊇]")

# A span smaller than this fraction of its line's dominant size, AND offset from the
# dominant baseline by more than SCRIPT_Y_TOL points, is a superscript/subscript rather
# than just a smaller symbol. Calibrated against real spans: body text is CMMI10/CMR10
# at 9.96pt, superscript/subscript runs are CMMI7/CMR7 at 6.97pt (ratio 0.70).
SCRIPT_SIZE_RATIO = 0.85
SCRIPT_Y_TOL = 1.0


@dataclass
class Span:
    text: str
    font: str
    size: float
    x0: float
    y0: float
    x1: float
    y1: float
    origin_y: float


@dataclass
class Line:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    fonts: frozenset[str]
    spans: tuple[Span, ...]


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _page_lines(page: pymupdf.Page) -> list[Line]:
    lines: list[Line] = []
    for block in page.get_text("dict")["blocks"]:
        for pdf_line in block.get("lines", []):
            raw_spans = pdf_line["spans"]
            text = "".join(s["text"] for s in raw_spans).strip()
            if not text:
                continue
            x0, y0, x1, y1 = pdf_line["bbox"]
            fonts = frozenset(s["font"] for s in raw_spans)
            spans = tuple(
                Span(
                    s["text"],
                    s["font"],
                    s["size"],
                    s["bbox"][0],
                    s["bbox"][1],
                    s["bbox"][2],
                    s["bbox"][3],
                    s["origin"][1],
                )
                for s in raw_spans
            )
            lines.append(Line(x0, y0, x1, y1, text, fonts, spans))
    lines.sort(key=lambda ln: (round(ln.y0, 1), ln.x0))
    return lines


def _has_font(fonts: frozenset[str], prefixes: tuple[str, ...]) -> bool:
    return any(font.startswith(p) for font in fonts for p in prefixes)


def _is_prose(line: Line, min_chars: int, math_frac_max: float = 0.3) -> bool:
    """A line is prose when it is long AND its characters are mostly non-math-font --
    NOT "zero math glyphs", which is too strict: an ordinary descriptive sentence like
    "of w(z) can be expressed in terms of w(z) and w'(z) as follows" is legitimate prose
    that merely names a variable inline, and a strict zero-math rule wrongly treated it as
    formula content, pulling unrelated sentences into a formula's walk. Conversely a true
    display formula built mostly from CMR-font function names ("cosh", "sinh") next to a
    couple of CMMI/CMSY symbols must still count as non-prose -- confirmed against the
    "| cosh x - cosh y| >= |x - y| ..." display, whose own char mix is only ~35% CMR."""
    if len(line.text) < min_chars:
        return False
    total = sum(len(s.text) for s in line.spans)
    if total == 0:
        return False
    math_chars = sum(
        len(s.text) for s in line.spans if _has_font(frozenset({s.font}), MATH_FONT_PREFIXES)
    )
    return (math_chars / total) < math_frac_max


def _has_x_overlap(fragments: list[Line], tol: float = 1.0) -> bool:
    """True if two fragments overlap horizontally by more than `tol` -- the signature of
    a vertically-stacked construct (fraction numerator/denominator) whose pieces happen
    to land within row_cluster_tolerance_pt of each other's y0 (a short numerator like
    "dw" sits close enough above its denominator's y0 to slip past row-clustering alone).
    Genuine same-line fragments tile left-to-right and never overlap this much."""
    ordered = sorted(fragments, key=lambda ln: ln.x0)
    return any(b.x0 < a.x1 - tol for a, b in zip(ordered, ordered[1:], strict=False))


def _row_is_contiguous(row: list[Line], max_gap: float) -> bool:
    """True if consecutive (x-sorted) fragments in a same-row group sit close enough to
    actually be adjacent words on one line, not two unrelated pieces that only coincide
    in y. Found necessary on NIST p.142 (4.35.40): a big multi-line fraction display's
    numerator/denominator sat outside walk_gap_pt in both directions (so neither was
    collected), leaving only the equation's "prefix =" and a stray trailing "." from the
    closing bracket -- 111pt apart, on the same row by y0 alone -- which the old
    unconditional " ".join blindly glued into "| tanh z| = .", silently dropping the
    entire fraction rather than skipping the id."""
    ordered = sorted(row, key=lambda ln: ln.x0)
    return all(b.x0 - a.x1 <= max_gap for a, b in zip(ordered, ordered[1:], strict=False))


def _row_clusters(fragments: list[Line], tol: float) -> list[list[Line]]:
    """Group fragments into physical text rows by y0, so a formula's neighbor content can
    be judged "one row" (safe to join) vs "several rows" (a stacked construct)."""
    rows: list[list[Line]] = []
    for frag in sorted(fragments, key=lambda ln: ln.y0):
        if rows and abs(frag.y0 - rows[-1][-1].y0) <= tol:
            rows[-1].append(frag)
        else:
            rows.append([frag])
    return rows


def _markup_spans(spans: list[Span]) -> str:
    """Reconstruct ^{}/_{} markup from raw glyph spans. Plain PDF text extraction has no
    marker for superscript/subscript -- a smaller, vertically-offset span IS the only
    signal available -- so without this every exponent/index flattens to plain characters
    ("x^n" -> "xn"), which is not LaTeX a fine-tune target should learn to produce."""
    if not spans:
        return ""
    normal_size = max(s.size for s in spans)
    baseline_candidates = [s.origin_y for s in spans if s.size >= normal_size - 0.1]
    baseline = statistics.median(baseline_candidates) if baseline_candidates else spans[0].origin_y

    groups: list[tuple[str, list[Span]]] = []
    for s in spans:
        if s.size < normal_size * SCRIPT_SIZE_RATIO:
            if s.origin_y < baseline - SCRIPT_Y_TOL:
                mode = "sup"
            elif s.origin_y > baseline + SCRIPT_Y_TOL:
                mode = "sub"
            else:
                mode = "normal"
        else:
            mode = "normal"
        if groups and groups[-1][0] == mode:
            groups[-1][1].append(s)
        else:
            groups.append((mode, [s]))

    out: list[str] = []
    for mode, grp in groups:
        text = "".join(s.text for s in grp)
        if mode == "normal":
            out.append(text)
            continue
        stripped = text.strip()
        if not stripped:
            out.append(text)
            continue
        if set(stripped) <= {"′", "″"}:
            # A bare prime is conventionally written "f'", not "f^{\prime}" -- both are
            # valid LaTeX, but this matches how a human would actually type it.
            out.append(stripped.replace("′", "'").replace("″", "''"))
            continue
        marker = "^" if mode == "sup" else "_"
        out.append(f"{marker}{{{stripped}}}")
    return "".join(out).strip()


def _fraction_bars(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Thin horizontal vector strokes on the page -- the actual fraction vinculum,
    confirmed against a real \\frac{dw}{dz} display (a zero-height stroke exactly between
    the numerator and denominator, spanning their shared width). Detecting the real ink
    is far safer than inferring a fraction from text geometry alone.

    width >= 8.0 specifically excludes the overline accent on a bar-conjugate variable
    (z-bar, "z̄") -- pdfTeX draws that as the exact same kind of zero-height stroke, only
    ~5pt wide (just spanning the one glyph). Confirmed on NIST p.31: a z̄ overline sat
    geometrically between two DIFFERENT equations' bodies and was picked up as if it were
    the vinculum joining them into one bogus fraction. A real vinculum was measured at
    ~12.6pt even over a single-character numerator ("d"), so 8pt separates the two
    cleanly with margin."""
    bars = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width >= 8.0 and r.height <= 1.5:
            bars.append(r)
    return bars


def _split_by_bar(
    line: Line, bar: pymupdf.Rect, tol: float = 3.0
) -> tuple[list[Span], list[Span], list[Span]]:
    """Split a line's spans into (before the bar, under the bar, after the bar) by each
    span's x midpoint -- a formula continuing past a fraction ("... + a^2w^2 = 1,") sits
    on the SAME pymupdf line as the denominator, not a separate one, so the denominator
    itself has to be picked out by position, not assumed to be the whole line."""
    before, under, after = [], [], []
    for s in line.spans:
        mid = (s.x0 + s.x1) / 2
        if mid < bar.x0 - tol:
            before.append(s)
        elif mid > bar.x1 + tol:
            after.append(s)
        else:
            under.append(s)
    return before, under, after


def _find_bar(num: Line, den: Line, bars: list[pymupdf.Rect]) -> pymupdf.Rect | None:
    """A bar must sit x-contained within num/den's shared width, and y-between the TOP of
    the numerator and the BOTTOM of the denominator. Deliberately not "num.y1 <= den.y0"
    (numerator strictly above, non-overlapping denominator): real glyph bboxes include
    ascender/descender padding, so a short numerator like "d" and its denominator's bbox
    routinely overlap in y even though the bar between them is real -- confirmed against
    the actual d/dz derivative table (NIST p.141), where this exact case was wrongly
    rejected by a same-way-too-strict non-overlap check before being loosened here."""
    x_lo = max(num.x0, den.x0) - 4
    x_hi = min(num.x1, den.x1) + 4
    if x_hi <= x_lo:
        return None
    y_lo, y_hi = num.y0, den.y1
    for bar in bars:
        mid_y = (bar.y0 + bar.y1) / 2
        if y_lo <= mid_y <= y_hi and bar.x0 >= x_lo and bar.x1 <= x_hi:
            return bar
    return None


def _try_fraction(num: Line, den: Line, bars: list[pymupdf.Rect]) -> tuple[str, str, str] | None:
    """Returns (prefix, frac_latex, tail) only when a real vinculum confirms num/den as a
    genuine fraction -- never constructed from glyph geometry alone."""
    bar = _find_bar(num, den, bars)
    if bar is None:
        return None
    if _has_font(num.fonts, UNRELIABLE_FONT_PREFIXES) or _has_font(
        den.fonts, UNRELIABLE_FONT_PREFIXES
    ):
        return None
    num_before, num_under, num_after = _split_by_bar(num, bar)
    if num_before or num_after:
        # The numerator's own line carries extra content beyond the bar's width (a
        # prefix like "t^{µ-1} * t^{-s-α} = " sharing the numerator's baseline, seen on
        # NIST p.70 2.6.39) -- only the denominator side's prefix/tail split (below) is
        # supported; a numerator-side prefix/tail is a more complex layout this doesn't
        # safely handle, so skip rather than risk splicing it in the wrong place.
        return None
    before, under, after = _split_by_bar(den, bar)
    if not under:
        return None  # nothing recognizable as the denominator itself -- don't guess
    num_text = _markup_spans(num_under)
    den_text = _markup_spans(under)
    if not num_text or not den_text:
        return None
    prefix = _markup_spans(before)
    tail = _markup_spans(after)
    if any(t.startswith(("^{", "_{")) for t in (prefix, tail, num_text, den_text)):
        # A superscript/subscript's base character landed on the OTHER side of the
        # bar-x-range boundary from its own script (both close enough to the boundary
        # that each was independently classified across it) -- confirmed on NIST p.125
        # 4.7.11, where the tail "z^a" split into denominator "...z" + orphaned "^{a}".
        # An orphaned ^{...}/_{...} with no base is a clear structural tell; don't guess
        # which side it belongs on, just drop the id.
        return None
    return prefix, f"\\frac{{{num_text}}}{{{den_text}}}", tail


def _extract_page_pairs(
    page: pymupdf.Page, pdf_page_1based: int, id_re: re.Pattern[str], cfg: dict
) -> list[dict]:
    lines = _page_lines(page)
    if not any(id_re.fullmatch(ln.text) for ln in lines):
        return []  # not a formula-bearing page (front matter, index, references, ...)

    bars = _fraction_bars(page)
    mid_x = page.rect.width / 2
    row_tol = cfg["row_cluster_tolerance_pt"]
    walk_gap = cfg["walk_gap_pt"]
    pad = cfg["crop_padding_pt"]
    max_chars = cfg["max_body_chars"]
    min_chars = cfg["prose_min_chars"]

    pairs: list[dict] = []
    for col_lines in (
        [ln for ln in lines if ln.x0 < mid_x],
        [ln for ln in lines if ln.x0 >= mid_x],
    ):
        col_lines.sort(key=lambda ln: ln.y0)
        for i, line in enumerate(col_lines):
            m = id_re.fullmatch(line.text)
            if not m:
                continue
            eqn_id = m.group(1)

            # Walk outward from this id, collecting the contiguous run of non-prose,
            # non-id lines immediately above and below it -- that run is this formula's
            # candidate content. A prose line, another id line, OR a y-gap bigger than
            # one text row (walk_gap_pt) ends the run -- the gap check is what stops a
            # neighboring formula's body (this book sometimes prints a formula's body
            # just ABOVE its own id, so the next id line alone is not always reachable
            # before the walk would otherwise wander into unrelated content).
            fragments: list[Line] = []
            last_y = line.y0
            j = i - 1
            while j >= 0:
                cand = col_lines[j]
                if id_re.fullmatch(cand.text) or _is_prose(cand, min_chars):
                    break
                if abs(cand.y0 - last_y) > walk_gap:
                    break
                fragments.append(cand)
                last_y = cand.y0
                j -= 1
            last_y = line.y0
            k = i + 1
            while k < len(col_lines):
                cand = col_lines[k]
                if id_re.fullmatch(cand.text) or _is_prose(cand, min_chars):
                    break
                if abs(cand.y0 - last_y) > walk_gap:
                    break
                fragments.append(cand)
                last_y = cand.y0
                k += 1

            if not fragments:
                continue  # id with no adjacent content at all -- nothing to pair
            rows = _row_clusters(fragments, row_tol)

            body_text: str | None = None
            content_lines: list[Line] = []

            if len(rows) == 1:
                row = sorted(rows[0], key=lambda ln: ln.x0)
                if len(row) >= 2 and _has_x_overlap(row):
                    if len(row) != 2:
                        continue  # more than 2 overlapping pieces -- not a simple fraction
                    num, den = sorted(row, key=lambda ln: ln.y0)
                    frac = _try_fraction(num, den, bars)
                    if frac is None:
                        continue  # overlapping but no confirmed vinculum -- don't guess
                    prefix, frac_latex, tail = frac
                    body_text = f"{prefix}{frac_latex}{tail}".strip()
                    content_lines = [num, den]
                else:
                    if len(row) > 1 and not _row_is_contiguous(row, cfg["max_frag_gap_pt"]):
                        continue  # same y, but too far apart in x to be one real line
                    if any(_has_font(ln.fonts, UNRELIABLE_FONT_PREFIXES) for ln in row):
                        continue
                    body_text = " ".join(_markup_spans(list(ln.spans)) for ln in row).strip()
                    content_lines = row
            elif len(rows) == 2:
                r0, r1 = rows
                if len(r0) != 1 or len(r1) != 1:
                    continue  # only the simple single-fragment num/den shape is supported
                num, den = sorted([r0[0], r1[0]], key=lambda ln: ln.y0)
                if not _has_x_overlap([num, den]):
                    continue  # two separate rows that aren't actually stacked -- ambiguous
                frac = _try_fraction(num, den, bars)
                if frac is None:
                    continue
                prefix, frac_latex, tail = frac
                body_text = f"{prefix}{frac_latex}{tail}".strip()
                content_lines = [num, den]
            else:
                continue  # more than 2 rows -- too complex, skip rather than scramble it

            if not body_text or len(body_text) > max_chars:
                continue
            if not RELATIONAL_RE.search(body_text):
                continue  # no relation symbol -- almost certainly a truncated capture
            if body_text.startswith(("^{", "_{")):
                continue  # orphaned superscript/subscript with no base -- malformed
            if body_text.lstrip().startswith("="):
                # A formula never legitimately starts with "=" -- confirmed on NIST
                # p.776 (34.3.14): the id labels only the tail of a genuinely multi-line
                # expression whose left-hand side is a different (unlabeled) line above
                # it, so even a "complete-looking" single row here is really a fragment.
                continue

            all_boxes = [line, *content_lines]
            x0 = min(ln.x0 for ln in all_boxes) - pad
            y0 = min(ln.y0 for ln in all_boxes) - pad
            x1 = max(ln.x1 for ln in all_boxes) + pad
            y1 = max(ln.y1 for ln in all_boxes) + pad
            clip = pymupdf.Rect(x0, y0, x1, y1) & page.rect

            pairs.append(
                {
                    "pair_id": f"nist_p{pdf_page_1based:04d}_{eqn_id.replace('.', '-')}",
                    "pdf_page": pdf_page_1based,
                    "eqn_id": eqn_id,
                    "text": f"{eqn_id}  {body_text}",
                    "clip": [clip.x0, clip.y0, clip.x1, clip.y1],
                }
            )
    return pairs


def run(cfg_path: str, limit_pages: int = 0) -> dict:
    cfg = _load_cfg(cfg_path)
    id_re = re.compile(cfg["eqn_id_pattern"])
    out_dir = Path(cfg["out_dir"])
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(cfg["source_pdf"])
    n_pages = doc.page_count if limit_pages <= 0 else min(limit_pages, doc.page_count)
    zoom = cfg["render_dpi"] / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    pairs_path = out_dir / "pairs.jsonl"
    written = 0
    skipped_dupes = 0
    pages_with_pairs = 0
    fraction_pairs = 0
    seen_ids: set[str] = set()
    t0 = time.time()
    with open(pairs_path, "w", encoding="utf-8") as out_fh:
        for i in range(n_pages):
            page = doc.load_page(i)
            pairs = _extract_page_pairs(page, i + 1, id_re, cfg)
            if not pairs:
                continue
            pages_with_pairs += 1
            for pair in pairs:
                # Belt-and-suspenders: a pair_id collision (same page + eqn_id matched
                # twice) would silently overwrite an already-written crop on disk while
                # both jsonl lines survive, leaving one line's "image" pointing at the
                # OTHER line's picture. eqn_id_pattern excluding parens (see config)
                # removes the one known cause; this refuses to write past it regardless.
                if pair["pair_id"] in seen_ids:
                    skipped_dupes += 1
                    continue
                seen_ids.add(pair["pair_id"])
                if "\\frac" in pair["text"]:
                    fraction_pairs += 1
                clip = pymupdf.Rect(*pair.pop("clip"))
                pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csGRAY, clip=clip)
                img_path = img_dir / f"{pair['pair_id']}.png"
                pix.save(img_path)
                pair["image"] = str(img_path).replace("\\", "/")
                out_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                written += 1
            if (i + 1) % 100 == 0:
                print(f"  page {i + 1}/{n_pages}: {written} pairs so far ({time.time() - t0:.0f}s)")
    doc.close()

    summary = {
        "pair_id_collisions_skipped": skipped_dupes,
        "source_pdf": cfg["source_pdf"],
        "pdf_pages_scanned": n_pages,
        "pages_with_pairs": pages_with_pairs,
        "pairs_written": written,
        "pairs_with_frac": fraction_pairs,
        "out_dir": out_dir.as_posix(),
        "render_dpi": cfg["render_dpi"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(cfg["summary_path"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", default="configs/nist_extract.yaml")
    p.add_argument("--limit-pages", type=int, default=0, help="0 = whole book")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run(args.cfg, args.limit_pages)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
