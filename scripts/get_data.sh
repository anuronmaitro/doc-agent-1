#!/usr/bin/env bash
# =============================================================================
# MathScholar (team 1) — fetch and render the corpus.
#
#   Abramowitz & Stegun, "Handbook of Mathematical Functions", NBS AMS-55 (1964)
#   Internet Archive item: handbookofmathem1964abra
#
# What this does, in order:
#   1. downloads the source PDF into  data/raw/   (skipped if already there)
#   2. renders every PDF page to      data/pages/ as 300-dpi GRAYSCALE PNG
#   3. spot-checks the printed<->PDF page offset against the PDF's text layer
#   4. prints the counts that data/provenance.md records
#
# Both output directories are gitignored — the corpus is ~1 GB and is never
# committed. This script is what recreates it on a clean machine.
#
# Idempotent and resumable: a page whose PNG already exists is skipped, so an
# interrupted run is fixed by re-running it.
#
# Usage:
#   bash scripts/get_data.sh                 # full corpus (~1082 pages)
#   LIMIT=20 bash scripts/get_data.sh        # first 20 pages only (quick check)
#   RENDER_DPI=150 bash scripts/get_data.sh  # lower-res pass
#   FORCE=1 bash scripts/get_data.sh         # re-render pages that already exist
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- what we are fetching -----------------------------------------------------
ARCHIVE_ID="handbookofmathem1964abra"
PDF_URL="https://archive.org/download/${ARCHIVE_ID}/${ARCHIVE_ID}.pdf"
RAW_DIR="data/raw"
PAGES_DIR="data/pages"
PDF_PATH="${RAW_DIR}/${ARCHIVE_ID}.pdf"

# Recorded at first download (2026-08-08). Re-runs verify against it so a
# truncated or silently-changed download cannot poison the corpus.
EXPECTED_SHA256="2e0205d8a35a0f544b7cd4f16fde91ba38f1527febedb37fbdc9b85a6e7b214a"

# --- render settings (config.yaml: ocr.render_dpi = 300) ----------------------
RENDER_DPI="${RENDER_DPI:-300}"

# Printed page N is PDF page N+32 — the book has 32 front-matter pages (cover,
# title, preface, contents) that carry no printed arabic folio.
FRONT_MATTER_OFFSET="${FRONT_MATTER_OFFSET:-32}"

LIMIT="${LIMIT:-0}"     # 0 = all pages
FORCE="${FORCE:-0}"     # 1 = re-render pages that already exist

echo "=============================================================="
echo " MathScholar corpus — Abramowitz & Stegun 1964 (NBS AMS-55)"
echo " public domain: work of the U.S. Government, 17 U.S.C. Sec.105"
echo "=============================================================="

# -----------------------------------------------------------------------------
# 0. Pick an interpreter that can import PyMuPDF.
#    We render with PyMuPDF rather than `pdftoppm` because poppler is not
#    installed by default on Windows, and pymupdf is already a pinned
#    dependency — so this script runs on all three of our machines unchanged.
# -----------------------------------------------------------------------------
# (an array, so a venv path containing spaces and the multi-word `uv run python`
#  fallback both survive quoting)
PY_CMD=()
for cand in "$REPO_ROOT/.venv/Scripts/python.exe" "$REPO_ROOT/.venv/bin/python" python3 python; do
  if "$cand" -c "import pymupdf" >/dev/null 2>&1; then PY_CMD=("$cand"); break; fi
done
if [ ${#PY_CMD[@]} -eq 0 ]; then
  if uv run python -c "import pymupdf" >/dev/null 2>&1; then
    PY_CMD=(uv run python)
  else
    echo "ERROR: PyMuPDF not available. Run:  uv sync --extra dev" >&2
    exit 1
  fi
fi
echo "[env] renderer: $("${PY_CMD[@]}" -c 'import pymupdf,sys; print("PyMuPDF", pymupdf.__doc__.split()[1], "on Python", sys.version.split()[0])')"

# -----------------------------------------------------------------------------
# 1. Download the PDF (78.6 MB). Resumable; skipped when already present.
# -----------------------------------------------------------------------------
mkdir -p "$RAW_DIR" "$PAGES_DIR"

if [ -s "$PDF_PATH" ]; then
  echo "[1/4] source PDF already present — skipping download ($PDF_PATH)"
else
  echo "[1/4] downloading $PDF_URL"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -C - -o "$PDF_PATH" "$PDF_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$PDF_PATH" "$PDF_URL"
  else
    echo "ERROR: need curl or wget to download the corpus." >&2
    exit 1
  fi
fi

# Integrity check — a half-finished download is the classic silent corpus bug.
ACTUAL_SHA256="$("${PY_CMD[@]}" -c "
import hashlib,sys
h=hashlib.sha256()
with open(sys.argv[1],'rb') as f:
    for b in iter(lambda: f.read(1<<20), b''): h.update(b)
print(h.hexdigest())
" "$PDF_PATH")"
echo "[1/4] sha256: $ACTUAL_SHA256"
if [ -n "$EXPECTED_SHA256" ] && [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "WARNING: sha256 differs from the recorded value in this script." >&2
  echo "         recorded: $EXPECTED_SHA256" >&2
  echo "         got:      $ACTUAL_SHA256" >&2
  echo "         Delete $PDF_PATH and re-run if the download was interrupted." >&2
fi

# -----------------------------------------------------------------------------
# 2-4. Render, spot-check the page offset, and report counts.
# -----------------------------------------------------------------------------
RENDER_DPI="$RENDER_DPI" \
FRONT_MATTER_OFFSET="$FRONT_MATTER_OFFSET" \
PDF_PATH="$PDF_PATH" \
PAGES_DIR="$PAGES_DIR" \
LIMIT="$LIMIT" \
FORCE="$FORCE" \
"${PY_CMD[@]}" - <<'PY'
import os
import re
import sys
import time

import pymupdf

pdf_path = os.environ["PDF_PATH"]
pages_dir = os.environ["PAGES_DIR"]
dpi = int(os.environ["RENDER_DPI"])
offset = int(os.environ["FRONT_MATTER_OFFSET"])
limit = int(os.environ["LIMIT"])
force = os.environ["FORCE"] == "1"


def page_stem(pdf_index_1based: int) -> str:
    """Filename stem for a PDF page.

    The number in the stem is the PRINTED page number, because that is what we
    cite and what every page list in summary.md is written in (see
    summary.md 3f: `as_p0255` -> printed page 255).

    Front-matter pages have no printed arabic folio, so they get an `as_f`
    stem keyed by PDF index instead. `ingest/loader.py` drops them.
    """
    printed = pdf_index_1based - offset
    if printed >= 1:
        return f"as_p{printed:04d}"
    return f"as_f{pdf_index_1based:04d}"


doc = pymupdf.open(pdf_path)
n_pdf_pages = doc.page_count
last = n_pdf_pages if limit <= 0 else min(limit, n_pdf_pages)

print(f"[2/4] rendering {last} of {n_pdf_pages} PDF pages "
      f"@ {dpi} dpi grayscale -> {pages_dir}/")
print(f"      naming: printed page N -> as_pNNNN.png   (printed N = PDF N+{offset})")

rendered = skipped = 0
t0 = time.time()
for i in range(last):
    out = os.path.join(pages_dir, page_stem(i + 1) + ".png")
    if os.path.exists(out) and not force:
        skipped += 1
        continue
    # csGRAY: the scan is monochrome typeset; grayscale keeps thin sub/superscript
    # strokes that hard binarisation would thicken, at 1/3 the disk of RGB.
    pix = doc.load_page(i).get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    pix.save(out)
    rendered += 1
    if rendered % 50 == 0:
        done = rendered + skipped
        rate = rendered / max(time.time() - t0, 1e-9)
        print(f"      {done}/{last} pages ({rate:.1f} pages/s)", flush=True)

print(f"[2/4] done: {rendered} rendered, {skipped} already present")

# --- 3. spot-check the printed <-> PDF offset --------------------------------
# Read the printed folio out of the PDF's own text layer on a few sample pages
# and confirm it equals (pdf_index - offset). This is the check that stops a
# whole downstream sprint from citing pages 32 off.
print(f"[3/4] verifying the page offset (expect printed N = PDF N+{offset})")


def folio_found(text: str, expected: int) -> bool:
    """True if `expected` appears as a running folio on this page.

    The folio sits at the very start or very end of the page's text, either
    alone ("260") or fused to the running head ("1028  LAPLACE TRANSFORMS").
    Anchoring to the line edge is what keeps this from matching the equation
    numbers and table cells that fill these pages.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:3] + lines[-3:]:
        if re.match(rf"^\W*{expected}\b", ln) or re.search(rf"\b{expected}\W*$", ln):
            return True
    return False


samples = [s for s in (255, 287, 292, 300, 632, 1060) if s <= n_pdf_pages]
confirmed = sum(folio_found(doc.load_page(p - 1).get_text(), p - offset) for p in samples)
# Control: the same test with a deliberately wrong offset must NOT pass, or the
# check is matching noise rather than folios.
control = sum(folio_found(doc.load_page(p - 1).get_text(), p) for p in samples)
for p in samples:
    hit = folio_found(doc.load_page(p - 1).get_text(), p - offset)
    print(f"      PDF p.{p:>4} -> printed {p - offset:>4}  {'OK' if hit else '-- (no folio in text layer)'}")
print(f"      confirmed {confirmed}/{len(samples)}; control at wrong offset {control}/{len(samples)}")

if confirmed == 0 or control >= confirmed:
    print(f"      WARNING: could not confirm printed N = PDF N+{offset} from the text layer.")
    print("      Open data/pages/as_p0255.png — it must be the Gamma-function page 255.")

# --- 4. report the numbers that go into data/provenance.md -------------------
pngs = sorted(f for f in os.listdir(pages_dir) if f.endswith(".png"))
content = [f for f in pngs if f.startswith("as_p")]
front = [f for f in pngs if f.startswith("as_f")]
bytes_pdf = os.path.getsize(pdf_path)
bytes_png = sum(os.path.getsize(os.path.join(pages_dir, f)) for f in pngs)

# Reference word count from the ARCHIVE's own OCR text layer. This is NOT the
# number validate() asserts: the >=60,000-word floor must come from OUR OCR
# (summary.md 10). It is recorded only to show the corpus is large enough.
words = sum(len(doc.load_page(i).get_text().split()) for i in range(n_pdf_pages))

print("[4/4] corpus summary  (these are the numbers in data/provenance.md)")
print(f"      PDF pages            : {n_pdf_pages}")
print(f"      page images rendered : {len(pngs)}  ({len(content)} printed + {len(front)} front matter)")
print(f"      printed page range   : 1 .. {n_pdf_pages - offset}")
print(f"      source PDF size      : {bytes_pdf / 1e6:.1f} MB")
print(f"      page images size     : {bytes_png / 1e9:.2f} GB")
print(f"      words (archive OCR)  : {words}   <- reference only, not the graded count")
print(f"      render dpi / mode    : {dpi} / grayscale")
doc.close()

if limit <= 0 and len(pngs) != n_pdf_pages:
    print(f"ERROR: expected {n_pdf_pages} page images, found {len(pngs)}", file=sys.stderr)
    sys.exit(1)
PY

echo
echo "Corpus ready. Nothing here is committed:"
echo "  $RAW_DIR/    source PDF      (gitignored)"
echo "  $PAGES_DIR/  page images     (gitignored)"
echo "Next: scripts/build_index.sh turns these into the vector index."
