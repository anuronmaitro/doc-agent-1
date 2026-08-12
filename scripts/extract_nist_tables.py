"""Stage C — NIST Handbook TABLE extraction (table-focused continuation fine-tune).

Requested directly (not in the original plan.md Step 25-30 sequence): after Step 28's
learning curve showed the fine-tuned reader working well on formulas/prose but hallucinating
on dense numeric tables (`as_p0509`, `as_p0351` -- see plan.md Step 28's writeup), extract
(image, text) TABLE pairs from the same NIST Handbook of Mathematical Functions (2010) PDF
`scripts/extract_nist_pairs.py` already uses for formula pairs, to give the model real
table-reading exposure Stage A structurally never had (Step 25 extracted formula crops only).

Reuses that script's real, position-based extraction discipline (real geometry, confirmed
against actual rendered pages, not guessed) rather than inventing a different standard:
`_page_lines`-style row clustering, the same "confirm before trusting geometry" ethos.

What real NIST tables actually look like (measured, not assumed -- p.97's Gauss-Laguerre
tables): a "Table N.M[.K]: caption text" line, then a short header row (e.g. "xk  wk"),
then data rows where each cell's fragments cluster into a small number of well-separated
x-position BANDS -- p.97 Table 3.5.6's column 1 fragments all sit at x0 in [175.9, 245.5],
column 2 at x0 in [315.6, 378.9], a clean ~70pt gap between them with zero overlap. Column
detection here is exactly that gap-clustering, generalized to N columns instead of
`extract_nist_pairs.py`'s fixed left/right page-column split.

Deliberately conservative, same reason as Step 25: a table is accepted only when every
included row's fragment count matches the header's column count exactly -- a row that
doesn't is DROPPED from that table, not guessed at, and a table left with under
MIN_DATA_ROWS accepted rows is skipped entirely. Complex multi-line "formula pair" tables
(e.g. p.46's Fourier-transform table, piecewise-function cells spanning 2-3 physical text
rows per logical row) are NOT what this targets -- those fail the row/column-count
consistency check and are skipped, which is correct: guessing their structure risks the
exact silent-wrong-label failure Step 25 was rewritten to avoid. This intentionally leaves
volume on the table for a future pass; see the module docstring's own precedent (Step 25's
695-pairs-not-6000 DECISION) for why "skip rather than scramble" beats chasing volume here.

Output `text` format matches the project's OWN established convention for A&S table
annotations (`data/README.md`'s OCR conventions, e.g. `data/annot/train/as_p1000.json`'s
Table 27.3) -- plain header + data rows, columns joined by two spaces -- NOT
`\begin{tabular}` LaTeX markup, so Stage C's target format is consistent with what Stage B
already trained the model to produce for tables, not a second, different convention.

Usage:
    python scripts/extract_nist_tables.py --cfg configs/nist_extract.yaml
    python scripts/extract_nist_tables.py --cfg configs/nist_extract.yaml --limit-pages 200
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import yaml

TABLE_CAPTION_RE = re.compile(r"^Table\s+(\d{1,2}\.\d{1,2}(?:\.\d{1,3})?)\s*:?\s*(.*)$")
# A table caption or a numbered-equation id (same pattern extract_nist_pairs.py uses) both
# end this table's body -- the equation-id case matters because a table is sometimes
# followed directly by prose containing the next display formula with no other boundary.
SECTION_BREAK_RE = re.compile(r"^(Table\s+[\d.]+|§[\d.]+|\d{1,2}\.\d{1,2}\.\d{1,3})\b")

ROW_TOL_PT = 2.5  # two fragments within this y0 gap are the same physical row
COLUMN_GAP_PT = 30.0  # an x-gap at least this wide between fragments marks a NEW column
# band -- measured on p.97: real inter-column gaps are ~70-110pt,
# intra-value gaps (e.g. within "0.26356 03197 18141") are <10pt.
MIN_DATA_ROWS = 3  # a "table" with fewer accepted rows than this isn't worth a pair
MAX_TABLE_ROWS = 15  # caps a single pair's row count -- keeps crops formula-pair-sized
# (Step 27's own finding: an extreme aspect-ratio Stage-A crop
# already loses effective resolution; a whole long table crammed
# into one crop would be worse) rather than one giant tall image.
MAX_ROWS_SCANNED_BELOW_CAPTION = 200  # safety bound; real tables never come close
PROBE_ROW_COUNT = 3  # rows used to establish column bands -- the first few rows right
# after the header, almost certainly genuine table content
CAPTION_X_MARGIN_PT = 60.0  # a fragment further left than this, relative to the caption's
# own x0, is unrelated running prose, not part of the table


@dataclass
class Frag:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _page_frags(page: pymupdf.Page) -> list[Frag]:
    """One Frag per pymupdf text LINE (not per span) -- a table cell's own internal spans
    (e.g. a value split across a size change) are joined here, same granularity
    `extract_nist_pairs.py`'s `_page_lines` uses, so column-gap clustering operates on
    whole cell values, not sub-glyph-run fragments."""
    frags: list[Frag] = []
    for block in page.get_text("dict")["blocks"]:
        for pdf_line in block.get("lines", []):
            text = "".join(s["text"] for s in pdf_line["spans"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = pdf_line["bbox"]
            frags.append(Frag(text, x0, y0, x1, y1))
    # Rounded to the nearest whole point, not 0.1pt -- see _extract_page_tables' comment
    # for the real case (two side-by-side table captions at y0=87.01/87.05) this fixes.
    frags.sort(key=lambda f: (round(f.y0), f.x0))
    return frags


def _row_clusters(frags: list[Frag], tol: float) -> list[list[Frag]]:
    rows: list[list[Frag]] = []
    for f in sorted(frags, key=lambda f: f.y0):
        if rows and abs(f.y0 - rows[-1][-1].y0) <= tol:
            rows[-1].append(f)
        else:
            rows.append([f])
    return rows


def _column_bands(body_rows: list[list[Frag]]) -> list[tuple[float, float]] | None:
    """Cluster every BODY-row fragment's x0 (deliberately excluding the header) into column
    bands by gap. Header-only: skipped because a short header label ("xk") is typically
    centered/offset relative to the WIDE numeric column beneath it -- measured on p.97
    Table 3.5.6, "xk"'s x0 (245.5) sits ~50pt right of its own column's data x0 range
    ([195.56, 195.72]), which would (and, before this fix, did) split gap-clustering into
    4 bands instead of 2 if the header were pooled in. Data rows across many rows still
    land in the same narrow x0 range (measured: that same column holds [195.56, 195.72]
    across 5 rows), so body-only pooling gives a reliable, header-independent column
    layout; the header is matched onto those bands separately (see `_assign_row_to_columns`,
    by fragment CENTER, not x0, for the same offset-tolerance reason). Returns None if
    fewer than 2 bands are found (not a multi-column table)."""
    xs = sorted(f.x0 for row in body_rows for f in row)
    if not xs:
        return None
    bands: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - bands[-1][-1] > COLUMN_GAP_PT:
            bands.append([x])
        else:
            bands[-1].append(x)
    if len(bands) < 2:
        return None
    return [(min(b), max(b)) for b in bands]


def _assign_row_to_columns(row: list[Frag], bands: list[tuple[float, float]]) -> list[str] | None:
    """Assign each fragment in `row` to its nearest column band by CENTER x (not x0) --
    matching by center, not left edge, is what lets a narrow, offset header label like
    "xk" still land in the same band as its wide data column (see `_column_bands`'s
    docstring for the measured offset). Returns None (reject this row) if any two
    fragments land in the SAME band -- a real row has exactly one value per column; two
    fragments claiming one band means either a wrapped/split value this simple model
    doesn't handle, or a misdetected band -- don't guess which, drop the row (Step 25's
    own "skip rather than scramble" rule, applied per-row here)."""
    cells: list[str | None] = [None] * len(bands)
    for f in sorted(row, key=lambda f: f.x0):
        center = (f.x0 + f.x1) / 2
        band_idx = min(
            range(len(bands)),
            key=lambda i: abs(center - (bands[i][0] + bands[i][1]) / 2),
        )
        if cells[band_idx] is not None:
            return None
        cells[band_idx] = f.text
    if any(c is None for c in cells):
        return None
    return cells  # type: ignore[return-value]


def _extract_table_at(
    frags_below: list[Frag], caption_frag: Frag, cfg: dict
) -> tuple[str, pymupdf.Rect] | None:
    """Given the fragments below a table caption on the same page, try to build a clean
    (header + up to MAX_TABLE_ROWS data rows) pair. Returns (text, bbox) or None.

    Where the table ENDS is decided by the table's own structure, not by recognizing what
    comes after it: a regex-based "next section" sniff (tried first, see SECTION_BREAK_RE's
    remaining use as a coarse pre-filter below) missed real cases -- p.97 Table 3.5.7 is
    immediately followed by a prose subheading ("Gauss Formula for a Logarithmic Weight
    Function") that matches none of "next Table caption / §N.N / bare eqn-id", so the row
    pool silently absorbed the whole next subsection before column-banding ever ran,
    producing 6 bogus bands instead of 2. Fixed by using PROBE_ROWS (the first few rows
    right after the header, almost certainly real table content) to establish the column
    bands, then scanning forward and stopping at the FIRST row that doesn't fit them --
    the table's own column structure is what marks its own end.
    """
    candidate = frags_below[:MAX_ROWS_SCANNED_BELOW_CAPTION]
    stop_at = next(
        (i for i, f in enumerate(candidate) if SECTION_BREAK_RE.match(f.text)), len(candidate)
    )
    candidate = candidate[:stop_at]
    # Some tables are typeset floated to the right of unrelated running prose that
    # continues at nearly the same y-position -- measured on p.107 Table 3.8.2: prose
    # ("Whether or not f0 and f1 have opposite signs...", x0=63) sits within 0.5pt of y0
    # of the real table row ("0  1.50000...", x0=399), so ROW_TOL_PT alone would merge
    # them into one bogus row. The caption's own x0 (354.7) sits close to the table's real
    # content (399-451), far from the unrelated prose (63) -- filtering to fragments no
    # more than CAPTION_X_MARGIN_PT left of the caption excludes that prose before row-
    # clustering ever runs, rather than trying to detect and strip it afterward.
    candidate = [f for f in candidate if f.x0 >= caption_frag.x0 - CAPTION_X_MARGIN_PT]
    if not candidate:
        return None

    rows = _row_clusters(candidate, ROW_TOL_PT)
    if len(rows) < 1 + MIN_DATA_ROWS:
        return None

    header_row, *body_rows = rows
    probe_rows = body_rows[:PROBE_ROW_COUNT]
    bands = _column_bands(probe_rows)
    if bands is None:
        return None

    header_cells = _assign_row_to_columns(header_row, bands)
    if header_cells is None:
        return None

    accepted: list[list[str]] = []
    accepted_frags: list[Frag] = list(header_row)
    for row in body_rows:
        if len(accepted) >= MAX_TABLE_ROWS:
            break
        cells = _assign_row_to_columns(row, bands)
        if cells is None:
            break  # first row that doesn't fit the established columns -- table ends here
        accepted.append(cells)
        accepted_frags.extend(row)

    if len(accepted) < MIN_DATA_ROWS:
        return None

    lines = [f"{caption_frag.text}", "  ".join(header_cells)]
    lines.extend("  ".join(row) for row in accepted)
    text = "\n".join(lines)

    pad = cfg["crop_padding_pt"]
    all_frags = [caption_frag, *accepted_frags]
    x0 = min(f.x0 for f in all_frags) - pad
    y0 = min(f.y0 for f in all_frags) - pad
    x1 = max(f.x1 for f in all_frags) + pad
    y1 = max(f.y1 for f in all_frags) + pad
    return text, pymupdf.Rect(x0, y0, x1, y1)


# A caption that wraps onto a second physical line (e.g. p.66 "Table 2.5.1: Domains of
# convergence for Mellin trans-" / "forms.") -- a genuine continuation is short, ends the
# sentence (period), and is NOT itself another caption/section-break/tabular-looking row.
# Measured against 129 real captions: several wrap this way; absorbing at most one such
# line keeps the body scan from starting mid-sentence instead of at the real header row.
_CAPTION_CONTINUATION_RE = re.compile(r"^[a-z].{0,40}\.$")


def _extract_page_tables(page: pymupdf.Page, pdf_page_1based: int, cfg: dict) -> list[dict]:
    # NOT two-column like the A&S corpus -- checked first (page.rect.width=612, and real
    # prose lines here span x0~63 to x1~567, i.e. nearly the FULL width), so this book's
    # body text is single-column; only individual TABLES place their own data columns
    # side-by-side. A whole-page column split was tried and made things worse (it cut a
    # single table's own two data columns in half). The actual cause of the observed
    # caption-order anomaly (p.47: "Table 1.14.3" sorted before "Table 1.14.2") was
    # `_page_frags`'s own sort-key rounding (fixed there, not here): two captions at
    # y0=87.01/87.05 -- the same visual row, two small tables placed side by side -- rounded
    # to DIFFERENT 0.1pt buckets (87.0 vs 87.1), so the finer-than-necessary rounding split
    # what should have been one row-group. `_page_frags` now rounds to the nearest whole
    # point, comfortably below real line spacing (~10-12pt) so it can't merge distinct rows.
    frags = _page_frags(page)
    out: list[dict] = []
    for i, f in enumerate(frags):
        m = TABLE_CAPTION_RE.match(f.text)
        if not m:
            continue
        table_id = m.group(1)
        below = frags[i + 1 :]
        if below and _CAPTION_CONTINUATION_RE.match(below[0].text):
            below = below[1:]
        result = _extract_table_at(below, f, cfg)
        if result is None:
            continue
        text, rect = result
        clip = rect & page.rect
        if clip.is_empty or clip.width <= 0 or clip.height <= 0:
            continue
        out.append(
            {
                "pair_id": f"nisttbl_p{pdf_page_1based:04d}_{table_id.replace('.', '-')}",
                "pdf_page": pdf_page_1based,
                "table_id": table_id,
                "text": text,
                "clip": [clip.x0, clip.y0, clip.x1, clip.y1],
            }
        )
    return out


def run(cfg_path: str, limit_pages: int = 0) -> dict:
    cfg = _load_cfg(cfg_path)
    out_dir = Path(cfg["out_dir"]).parent / "nist_tables"  # data/annot/nist_tables/
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(cfg["source_pdf"])
    n_pages = doc.page_count if limit_pages <= 0 else min(limit_pages, doc.page_count)
    zoom = cfg["render_dpi"] / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    pairs_path = out_dir / "pairs.jsonl"
    written = 0
    pages_with_tables = 0
    seen_ids: set[str] = set()
    t0 = time.time()
    with open(pairs_path, "w", encoding="utf-8") as out_fh:
        for i in range(n_pages):
            page = doc.load_page(i)
            pairs = _extract_page_tables(page, i + 1, cfg)
            if not pairs:
                continue
            pages_with_tables += 1
            for pair in pairs:
                if pair["pair_id"] in seen_ids:
                    continue
                seen_ids.add(pair["pair_id"])
                clip = pymupdf.Rect(*pair.pop("clip"))
                pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csGRAY, clip=clip)
                img_path = img_dir / f"{pair['pair_id']}.png"
                pix.save(img_path)
                pair["image"] = str(img_path).replace("\\", "/")
                out_fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
                written += 1
            if (i + 1) % 100 == 0:
                print(
                    f"  page {i + 1}/{n_pages}: {written} table pairs so far "
                    f"({time.time() - t0:.0f}s)"
                )
    doc.close()

    summary = {
        "source_pdf": cfg["source_pdf"],
        "pdf_pages_scanned": n_pages,
        "pages_with_tables": pages_with_tables,
        "table_pairs_written": written,
        "out_dir": out_dir.as_posix(),
        "render_dpi": cfg["render_dpi"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
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
