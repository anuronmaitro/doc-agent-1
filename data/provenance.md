# Corpus provenance — MathScholar (team 1)

Everything below is **measured**, not estimated. Re-running `bash scripts/get_data.sh`
reprints the same numbers; that script is the single source of this file.

---

## Source

| | |
|---|---|
| **Work** | Milton Abramowitz & Irene A. Stegun (eds.), *Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables* |
| **Publisher / series** | U.S. National Bureau of Standards, **Applied Mathematics Series 55** (AMS-55), issued June 1964 |
| **Internet Archive item** | `handbookofmathem1964abra` — <https://archive.org/details/handbookofmathem1964abra> |
| **Direct file** | <https://archive.org/download/handbookofmathem1964abra/handbookofmathem1964abra.pdf> |
| **Alternate scan (not used)** | `AandS-mono600` — a tighter mono scan, kept as a fallback mirror only |
| **Retrieved** | **2026-08-08** (the day `scripts/get_data.sh` was first run for A2) |
| **sha256 of the PDF** | `2e0205d8a35a0f544b7cd4f16fde91ba38f1527febedb37fbdc9b85a6e7b214a` |

The checksum is asserted by `scripts/get_data.sh` on every run, so a truncated or
silently-changed download cannot enter the corpus unnoticed.

### What we take, and what we throw away

We use the **scanned page images only**. The archive PDF ships a low-quality
auto-OCR text layer; we **discard it** for all graded purposes and run our own
math-aware OCR in A2. Reading this scan is the graded work — using the bundled
text layer would be marking our own homework.

The one place the archive text layer is allowed is as a *size sanity check* and
as the input to the OCR-hardness ranking that picked our annotation pages
(`tools/hardness.py`). Neither feeds an accuracy number.

---

## Licence

- **Status: public domain.** AMS-55 is a work prepared by U.S. Government
  employees at the National Bureau of Standards, so it carries no copyright in
  the United States (17 U.S.C. §105).
- **Redistribution:** permitted freely, with attribution to NBS / Abramowitz &
  Stegun. We nevertheless ship `scripts/get_data.sh` rather than the bytes, so
  the grader recreates `data/raw/` and `data/pages/` from the original source.

---

## How to recreate the corpus

```bash
bash scripts/get_data.sh          # ~78.6 MB download, then ~1082 page renders
LIMIT=20 bash scripts/get_data.sh # quick check on the first 20 pages
```

It downloads the PDF to `data/raw/`, renders every page to `data/pages/`, verifies
the printed↔PDF page offset against the PDF's own text layer, and prints the
measured counts in this file. It is **idempotent and resumable** — a page whose
PNG already exists is skipped, so an interrupted run is fixed by re-running it.

Both directories are gitignored. **The corpus is never committed** (~0.9 GB); only
the held-out sample pages in `grading_kit/heldout_pages/` are.

---

## Size (measured 2026-08-08)

| Quantity | Value | How measured |
|---|---|---|
| PDF pages | **1082** | `pymupdf.open(...).page_count` |
| Content pages | **~1046** | 1082 minus 32 front-matter pages and the blank versos / pure-plate pages `ingest/loader.py` drops |
| Printed page range | **1 … 1050** | PDF 33…1082 at the offset below |
| Source PDF size | **78.6 MB** (78,607,704 bytes) | `os.path.getsize` |
| Rendered page images | **1082 PNG** (1050 `as_p*` + 32 `as_f*`), **0.94 GB** | `data/pages/` after a full run |
| Page geometry | **557.7 × 751.3 pt** → 2324 × 3131 px @ 300 dpi | `page.rect` |
| Render settings | **300 dpi, 8-bit grayscale** | matches `configs/config.yaml → ocr.render_dpi: 300` |
| Words in the archive OCR text layer | **579,798** | sum of `page.get_text().split()` over all 1082 pages |

**On the word count.** 579,798 is ~9.7× the 60,000-word floor, so the corpus is
comfortably large enough. Two caveats, both deliberate:

1. This counts the **archive's** OCR, not ours. The floor that
   `data/validate.py` asserts must be met by **our own OCR** after Stage 3
   (summary §10) — this number only establishes that the pages contain enough
   text to be worth reading.
2. It **supersedes the 417,277 reported in A1**, which used a different
   tokenisation. Cross-check: the archive's own full-text file (`fts.txt`)
   contains 580,463 words, within 0.1 % of our 579,798 — so 579,798 is the
   figure to quote.

---

## Page numbering — printed vs PDF (read this before citing anything)

> **printed N = PDF N+32**

The scan opens with 32 unnumbered front-matter pages (cover, title, preface,
contents), so the printed folio runs 32 behind the PDF index.

**We name every page image by its PRINTED page number**, because that is what a
citation has to say and what every page list in `summary.md §4h` is written in:

| Stem | Meaning | Example |
|---|---|---|
| `as_pNNNN.png` | printed page NNNN | `as_p0255.png` = printed 255 = PDF page 287 |
| `as_fNNNN.png` | front matter, keyed by PDF index | `as_f0001.png` = PDF page 1 (cover) |

1050 `as_p*` + 32 `as_f*` = 1082 files. `ingest/loader.py` (Step 8) drops the
`as_f*` pages plus the blank versos, leaving the ~1046 content pages.

**This corrects a real bug carried over from A1.** The A1 `get_data.sh` named
files by *PDF index* (`as_p{i:04d}` for PDF page `i`), which would have made
`as_p0255.png` the printed page 223 — while A1's own `labels.jsonl` records
`{"page_id": "as_p0255", "printed_page": 255}` and `summary.md §3f` maps
`as_p0255 → 255`. The A1 gold images are correct; the A1 script that claimed to
produce them was not. A2 uses the printed-page convention throughout, so the
three carried-over gold pages line up without renaming.

**Verification (not assumed — checked).** `get_data.sh` reads the printed folio
out of the PDF's text layer on six sampled pages and confirms it equals
`pdf_index − 32`, with a control run at the wrong offset that must fail:

```
PDF p. 255 -> printed  223  OK        # folio "223" on the page
PDF p. 287 -> printed  255  OK        # start of Ch.6, the Gamma function
PDF p. 292 -> printed  260  OK
PDF p. 300 -> printed  268  OK
PDF p. 632 -> printed  600  OK
PDF p.1060 -> printed 1028  OK
confirmed 6/6; control at wrong offset 0/6
```

**Second, independent check.** Our freshly rendered pages were compared against
the three A1 gold images pixel-wise. Image dimensions match exactly — including
the page-specific heights (3053 / 3046 / 3056 px) — and the correlation
separates the two candidate conventions cleanly:

| A1 gold page | vs **printed N** (our convention) | vs PDF-index N (A1 script's convention) |
|---|---|---|
| `as_p0243` | **0.758** | 0.079 |
| `as_p0255` | **0.830** | 0.065 |
| `as_p0360` | **0.851** | 0.070 |

So the A1 gold images are already on the printed-page convention and drop into
`grading_kit/heldout_pages/` unchanged at Step 5. (The correlation is below 1.0
because A1 rendered through a different rasteriser with different antialiasing,
not because the page differs.)

---

## Structure and split policy

**Structure.** 29 independently-authored chapters (Ch.1 Mathematical Constants …
Ch.6 Gamma … Ch.9 Bessel … Ch.29 Laplace Transforms). Each chapter mixes prose,
numbered formula blocks (`6.1.8`), and dense numeric tables, typeset in two
columns.

**Split unit: the chapter — 20 build / 4 val / 5 test of 29** (≈69 / 14 / 17 %).

- **Why the chapter, not the page.** The book is one continuous work, so the
  chapter is the unit of correlation: notation, typeface conventions and
  cross-references are shared *within* a chapter. A page-level split would leak
  that context across splits and inflate every score.
- **Chosen, not positional.** Test = the special-function chapters our users
  actually query (Ch.5 exponential integral, Ch.6 gamma, Ch.7 error/Fresnel,
  Ch.9 Bessel, Ch.17 elliptic integrals), so evaluation lands on real traffic.
  All three A1 gold pages (printed 243, 255, 360) fall in test chapters.
- **Val = Ch.8, 10, 13, 15**, reserved for OCR early-stopping and checkpoint
  selection in A2, and for fitting the confidence calibration of our Calibrated
  NFR in A3.
- **`doc_id` = chapter id** (e.g. `ch06_gamma`) — see `summary.md §3f`. Making
  the chapter the "document" is what lets `data/validate.py` assert *"split by
  document, not page"* directly.

**Leakage check.** `notebooks/eda.ipynb` asserts the build / val / test chapter
sets are **pairwise disjoint**, and that every gold page lies in a test chapter.
`data/validate.py` (Step 6) re-asserts disjointness by `doc_id` at pipeline time,
so the guarantee is enforced in code and not only in a notebook.

**Hand-verified pages (built in Sprint 3, listed in `summary.md §4h`; train count updated at
the Step 18b gate, `src/doc_agent/data/validate.py`):**
122 train (build chapters -- 105 from `summary.md §4h` + 17 added 2026-08-10 when the
repaired reader's full-book run measured a 28.5% no-output rate, above the Step 18b gate's
25% line, targeting the chapters it failed hardest on) + 20 validation (val chapters) + 39
test (test chapters) = **181 pages**, disjoint by construction because their source chapter
sets are disjoint.

---

## What we strip

- The bundled auto-OCR text layer (OCR must be our own graded work).
- The 32 front-matter pages, blank versos and pure-plate pages — dropped by
  `ingest/loader.py`, leaving ~1046 content pages.
- Running heads and footers are **not** deleted from the image; they are
  excluded at annotation time by the conventions in `data/README.md`, and the
  printed folio is kept as page metadata because it is what we cite.

---

## What makes this corpus hard (the A1 speciality, restated)

Dense mathematical notation on a 1964 letterpress scan: two-column layout,
multi-level sub/superscripts, wide integrals and radicals whose glyphs span
several lines, and page-filling numeric tables. Generic OCR fragments exactly
these regions — which is both why `math-notation` is our declared data
speciality and why the fine-tune in Sprint 4 has something real to fix.
