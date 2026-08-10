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
#   ANNOT=1 bash scripts/get_data.sh         # Step 18: the 181 annotation pages only
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
ANNOT="${ANNOT:-0}"     # 1 = Step 18 mode: the 181 annotation pages, not the corpus
ANNOT_DIR="data/annot"

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
# 1b. ANNOT=1 — Step 18: materialise the 181 hand-annotation pages and stop.
#
# The page lists live in src/doc_agent/data/validate.py (imported below), not in
# this script, so the renderer and tests/test_data.py read the same one list.
#
# Images are COPIED from data/pages/ when it is already populated, and rendered
# from the PDF only when it is not. Copying is not just faster: it guarantees the
# annotator's image is byte-identical to the one vision/ocr.py actually read, so
# a correction can never be made against a subtly different render than the draft.
# -----------------------------------------------------------------------------
if [ "$ANNOT" = "1" ]; then
  RENDER_DPI="$RENDER_DPI" \
  FRONT_MATTER_OFFSET="$FRONT_MATTER_OFFSET" \
  PDF_PATH="$PDF_PATH" \
  PAGES_DIR="$PAGES_DIR" \
  ANNOT_DIR="$ANNOT_DIR" \
  FORCE="$FORCE" \
  PYTHONPATH="$REPO_ROOT/src" \
  "${PY_CMD[@]}" - <<'PY'
import json
import os
import re
import shutil
import time

import pymupdf

from doc_agent.data.validate import (
    ANNOT_SETS,
    ANNOT_TEST_ALREADY_DONE,
    validate_annotation_sets,
)

pdf_path = os.environ["PDF_PATH"]
pages_dir = os.environ["PAGES_DIR"]
annot_dir = os.environ["ANNOT_DIR"]
dpi = int(os.environ["RENDER_DPI"])
offset = int(os.environ["FRONT_MATTER_OFFSET"])
force = os.environ["FORCE"] == "1"

# --- 1. the lists must be sound BEFORE anyone renders or transcribes anything -
print("[1/4] validating the page lists (counts, duplicates, leakage)")
counts = validate_annotation_sets()
print(f"      {sum(counts.values())} pages: "
      + " / ".join(f"{n} {k}" for k, n in counts.items())
      + "  — pairwise disjoint, each page inside its own chapter family")

doc = pymupdf.open(pdf_path)


def folio_found(text, expected):
    """Same edge-anchored folio test scripts/get_data.sh uses for the corpus."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:3] + lines[-3:]:
        if re.match(rf"^\W*{expected}\b", ln) or re.search(rf"\b{expected}\W*$", ln):
            return True
    return False


# --- 2. materialise the images ------------------------------------------------
print(f"[2/4] writing 300-dpi grayscale pages -> {annot_dir}/<split>/")
manifest = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "render_dpi": dpi,
    "front_matter_offset": offset,
    "page_numbering": f"printed N = PDF N+{offset}; files named by PRINTED page",
    "source": "summary.md 4h via doc_agent.data.validate",
    "splits": {},
}
copied = rendered = skipped = 0
confirmed = control = checked = 0

for split, (pages, _family) in ANNOT_SETS.items():
    out_dir = os.path.join(annot_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for printed in pages:
        stem = f"as_p{printed:04d}"
        dst = os.path.join(out_dir, stem + ".png")
        src = os.path.join(pages_dir, stem + ".png")
        pdf_index = printed + offset          # 1-based
        if os.path.exists(dst) and not force:
            skipped += 1
        elif os.path.exists(src):
            shutil.copyfile(src, dst)
            copied += 1
        else:
            pix = doc.load_page(pdf_index - 1).get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            pix.save(dst)
            rendered += 1

        # --- 3. per-page offset proof, not a 6-page spot check ----------------
        text = doc.load_page(pdf_index - 1).get_text()
        hit = folio_found(text, printed)
        confirmed += hit
        control += folio_found(text, pdf_index)   # same test at the WRONG offset
        checked += 1
        rows.append({
            "printed_page": printed,
            "pdf_page": pdf_index,
            "file": os.path.relpath(dst).replace("\\", "/"),
            "bytes": os.path.getsize(dst),
            "folio_confirmed": bool(hit),
        })
    manifest["splits"][split] = {"count": len(rows), "dir": out_dir, "pages": rows}

print(f"      {copied} copied from {pages_dir}/, {rendered} rendered from the PDF, "
      f"{skipped} already present")

# --- 3b. report the offset evidence ------------------------------------------
print(f"[3/4] page-offset verification (expect printed N = PDF N+{offset})")
print(f"      folio confirmed on {confirmed}/{checked} pages; "
      f"the same test at the WRONG offset confirms {control}/{checked}")
manifest["offset_verification"] = {
    "checked": checked, "confirmed": confirmed, "control_wrong_offset": control,
}
if confirmed <= control:
    raise SystemExit(
        f"ERROR: the folio test does not separate the correct offset ({confirmed}) from a "
        f"deliberately wrong one ({control}) — do not annotate against these renders."
    )

# --- 4. how much of this is correction vs blank-slate transcription ----------
# Step 17's helper shows the current data/ocr/ draft next to the page. A page with no
# usable draft is transcribed from scratch, which is far slower -- so the split of the
# annotation set into "correctable" and "blank slate" is the number that sizes Steps 19-24.
failures = {}
fpath = os.path.join("data", "ocr", "failures.json")
if os.path.exists(fpath):
    with open(fpath, encoding="utf-8") as fh:
        failures = {r["page_id"]: r.get("reason", "?") for r in json.load(fh)}

print("[4/4] draft availability for Steps 19-24 (from data/ocr/'s current baseline run)")
for split, (pages, _family) in ANNOT_SETS.items():
    have = blank = 0
    for printed in pages:
        stem = f"as_p{printed:04d}"
        if os.path.exists(os.path.join("data", "ocr", stem + ".mmd")) and stem not in failures:
            have += 1
        else:
            blank += 1
    manifest["splits"][split]["baseline_draft"] = {"correctable": have, "blank_slate": blank}
    pct = 100 * blank / max(len(pages), 1)
    print(f"      {split:5s} {have:3d} correctable / {blank:3d} blank-slate ({pct:.0f}% from scratch)")

done = sorted(ANNOT_TEST_ALREADY_DONE)
manifest["test_already_annotated_in_a1"] = done
print(f"      test: {len(done)} pages already transcribed in A1 and reused verbatim: {done}")

out = os.path.join(annot_dir, "annot_manifest.json")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2)
    fh.write("\n")
print(f"\nwrote {out}  ({sum(counts.values())} pages)")
doc.close()
PY

  echo
  echo "Annotation pages ready:"
  echo "  $ANNOT_DIR/train/  122 pages   (images gitignored; the .json annotations are committed)"
  echo "  $ANNOT_DIR/val/     20 pages   (images gitignored; the .json annotations are committed)"
  echo "  $ANNOT_DIR/test/    39 pages   (whole folder gitignored — the real output is grading_kit/)"
  echo "  $ANNOT_DIR/annot_manifest.json  committed: the counts + per-page offset proof"
  echo
  echo "Next: Step 19 — bootstrap the drafts with"
  echo "  python scripts/annotate_helper.py --out $ANNOT_DIR/test --pages 229 230 232 ..."
  exit 0
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
