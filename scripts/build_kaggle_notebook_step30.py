"""Generates KAGGLE/step30_full_ocr_reindex/kaggle_step30_full_ocr_reindex.ipynb from
run_step30.py (plan.md 11.7's naming convention: folder + `kaggle_<name>.ipynb` source +
`scripts/build_kaggle_notebook_step30.py` builder, matching step29's shape).

Embeds src/doc_agent/vision/ocr.py, configs/config.yaml, and run_step30.py verbatim via
`%%writefile` -- none of them are on `main` yet (the region-routing Reader integration,
Step 28 point 11, is still local-only), so Kaggle would otherwise clone the stale version.
Regenerate after changing any of the three:
    python scripts/build_kaggle_notebook_step30.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = REPO_ROOT / "KAGGLE" / "step30_full_ocr_reindex"
OCR_MODULE = (REPO_ROOT / "src" / "doc_agent" / "vision" / "ocr.py").read_text(encoding="utf-8")
CONFIG_YAML = (REPO_ROOT / "configs" / "config.yaml").read_text(encoding="utf-8")
RUN_STEP30 = (KAGGLE_DIR / "run_step30.py").read_text(encoding="utf-8")

REPO_URL = "https://github.com/anuronmaitro/doc-agent-1.git"
OWNER = "eliasmainur"
KERNEL_SLUG = "mathscholar-step30-full-ocr-reindex"
CKPT_DATASET = "eliasmainur/mathscholar-step30-ckpt"

# True once the timing question is settled. It is, for this push: OCR already completed on
# 2026-08-13 (15.26s/page, 4.4h) and is reseeded from CKPT_DATASET, so there are no GPU
# hours left to gate -- the run only has to finish chunk -> index -> validate. Set back to
# False if you ever regenerate this for a genuine from-scratch re-OCR, so the smoke
# projection gets reviewed before committing the hours.
INCLUDE_FULL_RUN = True

EMBEDDED_PATHS = (
    "src/doc_agent/vision/ocr.py",
    "configs/config.yaml",
)


def _check_embeds_complete() -> None:
    """Same guard build_kaggle_notebook.py uses (Step 28 point 8) -- fails loudly if any
    file under src/doc_agent/ or configs/ differs from `main` and isn't embedded above."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "main", "--", "src/doc_agent", "configs"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    changed = {ln.strip().replace("\\", "/") for ln in result.stdout.splitlines() if ln.strip()}
    result_u = subprocess.run(
        ["git", "status", "--porcelain", "-uall", "--", "src/doc_agent", "configs"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for ln in result_u.stdout.splitlines():
        if ln.startswith("??"):
            changed.add(ln[3:].strip().replace("\\", "/"))

    missing = [pth for pth in changed if pth not in EMBEDDED_PATHS]
    if missing:
        raise SystemExit(
            "build_notebook (step30): these files differ from main (or are untracked) and "
            "are NOT embedded in the notebook:\n  " + "\n  ".join(sorted(missing))
        )


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def build_cells() -> list[dict]:
    cells: list[dict] = []

    cells.append(md("""# MathScholar — Step 30: full-book re-OCR + index rebuild

Owner: Elias Mainur (S3). Runs `plan.md` Step 30: re-OCR the full book with the
fine-tuned reader and rebuild the vector index.

**Reader: `curve_n122` + the region-routing hybrid (Step 28 point 11)** — `table_ft`
applied only to detected table-region crops, curve_n122 whole-page everywhere else.
Step 29 has since CONFIRMED this choice (plan.md Step 30 point 1, ✅ RESOLVED 2026-08-13):
on the 39 real TEST pages the hybrid cut failure rate from 23.1% to 7.7%, flat on
char-F1/exact-match. No re-run needed on reader grounds.

**Corpus size:** the book renders to 1082 PDF pages = 1050 printed + 32 front matter;
`loader.load_pages()` drops front matter, blank versos and pure plates, leaving
**1040 content pages** measured. plan.md's "~1046" is `data/provenance.md`'s estimate —
report 1040 as the Section 5 denominator, since that is what actually loads.

> **This push is the index-rebuild completion of the 2026-08-13 run.** That run finished
> OCR (987/1040 pages, 53 failures, 4.4h) and embedding, then died in `store.build()` on
> `import faiss` — numpy had resolved to 2.5.1 against a faiss-cpu 1.8 binary built for
> the NumPy 1.x C-ABI. Fixed on `main` by pinning `numpy<2.0`. With `FRESH_START=False`
> and the checkpoint reseeded below, OCR and embeddings are cache hits, so this push
> should only pay for chunk/index/validate.

**Resumability**: every stage of `pipeline.build_knowledge_base()` already caches
(`data/ocr/<page_id>.mmd`, `data/interim/*.png`, `data/index/embed_cache.npz`) — a
killed/timed-out session's Output, re-attached as a Dataset on the next push, resumes
exactly where it left off. See **"Resuming after a timeout"** near the bottom.

**Do not edit this notebook directly on kaggle.com** — generated by
`scripts/build_kaggle_notebook_step30.py` from `run_step30.py` +
`src/doc_agent/vision/ocr.py` + `configs/config.yaml`. Change those, then regenerate."""))

    cells.append(
        code(
            f"""# Resumed push: the 2026-08-13 run's OCR + embeddings are reseeded from the
# checkpoint dataset below, so this push only has to finish chunk -> index -> validate.
# Set FRESH_START = True and RESEED_DATASET = None for a genuine from-scratch re-OCR.
FRESH_START = False
RESEED_DATASET = "{CKPT_DATASET}"
REPO_URL = "{REPO_URL}"
BRANCH = "main"
"""
        )
    )

    cells.append(md("""## 1. Clone the repo and install pinned dependencies"""))
    cells.append(code("""import os
import subprocess

if not os.path.exists("/kaggle/working/repo"):
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, "/kaggle/working/repo"], check=True)
%cd /kaggle/working/repo
!pip install -q --no-cache-dir -r requirements.lock
import pkg_resources  # noqa: F401

print("pkg_resources OK")
!df -h /kaggle/working
"""))

    cells.append(md("""## 1b. Clear stale baseline-reader OCR cache (FRESH_START only)

`ocr.transcribe()`'s resumability cache (`if mmd_path.exists(): skip the model`) cannot
tell WHICH reader produced a cached file -- it just sees one and reuses it. If any
pretrained-baseline `.mmd` were present at the start of a fresh run, those pages would be
kept as-is and the fine-tuned reader would only ever touch what was left, defeating the
entire point of this step. So on a genuine fresh start we clear the CLONE's working copy
(never the committed source).

As of commit `351b203` ("prep: preserve Step 18b baseline OCR ahead of Step 30's re-OCR")
this is belt-and-braces: the 744 baseline files that used to sit in `data/ocr/` were moved
to `data/old baseline ocr/`, so a fresh clone now arrives with `data/ocr/` empty and the
loop below clears 0 files. Kept anyway -- it costs nothing and it is the guard that makes
"fresh means fresh" true regardless of what a future clone happens to carry.

A real resume (FRESH_START=False) skips this: cell 2 reseeds from a PREVIOUS STEP 30
push's own checkpoint, which only ever contained fine-tuned output. Verified empirically
for the 2026-08-13 checkpoint -- all 709 pages overlapping the old baseline snapshot were
byte-different from it, i.e. genuinely re-transcribed, 0 stale carry-over."""))
    cells.append(code("""import shutil

if FRESH_START:
    n_cleared = 0
    if os.path.isdir("data/ocr"):
        for name in os.listdir("data/ocr"):
            if name.endswith(".mmd"):
                os.remove(os.path.join("data/ocr", name))
                n_cleared += 1
    print(f"FRESH_START: cleared {n_cleared} stale baseline-reader .mmd file(s) from data/ocr/")
    # meta.jsonl / failures.json are keyed by chunk_id / page_id and safely regenerated;
    # clearing them avoids stale confidence/failure rows for pages we're about to redo.
    for stale in ("data/ocr/meta.jsonl", "data/ocr/failures.json"):
        if os.path.exists(stale):
            os.remove(stale)
else:
    print("FRESH_START=False: leaving data/ocr/ as-is (expecting cell 2's reseed)")
"""))

    cells.append(md("""## 2. Resume from a previous push's checkpoint (skip if this is a fresh run)

If `RESEED_DATASET` is set and attached to this kernel (Add Input), copies its
`data/ocr/`, `data/interim/`, and `data/index/embed_cache.npz` into the fresh clone
BEFORE the run starts, so already-transcribed pages/embeddings are skipped rather than
redone. Self-diagnosing (same lesson as Stage C's mount cell, plan.md Step 28 point 10):
lists `/kaggle/input/` and searches by content rather than assuming one exact path."""))
    cells.append(code("""# shutil already imported in the "clear stale cache" cell above.
if not FRESH_START and RESEED_DATASET:
    print("Available /kaggle/input/ entries:", os.listdir("/kaggle/input"))
    # Find the checkpoint root: the directory that directly contains "ocr"/"index" as
    # SIBLING subdirectories (matching how "kaggle datasets create -p ./out/data" uploads
    # -- dataset root IS the former "data/" folder, so its children are ocr/, interim/,
    # index/ with no "data/" prefix). Binding src_root to the "ocr" leaf itself (the
    # directory whose OWN files match) was the original, unexercised bug here: the copy
    # loop below then re-appended "data/ocr" onto a path that was already the ocr folder.
    #
    # Walk ALL of /kaggle/input, never a guessed top-level name. Kaggle has mounted this
    # same attached dataset two different ways: /kaggle/input/mathscholar-step30-ckpt/ on
    # the 2026-08-13 push, and /kaggle/input/datasets/<...>/ on the re-push, where the only
    # top-level entry is the literal "datasets". Matching a slug against the top level
    # therefore found nothing and aborted a resume whose data was present all along. The
    # cell already claimed to "search by content rather than assuming one exact path"; this
    # is that claim actually implemented.
    src_root = None
    for root, dirs, _files in os.walk("/kaggle/input"):
        ocr_dir = os.path.join(root, "ocr")
        index_dir = os.path.join(root, "index")
        if "ocr" in dirs and any(f.endswith(".mmd") for f in os.listdir(ocr_dir)):
            src_root = root
            break
        if "index" in dirs and os.path.exists(os.path.join(index_dir, "embed_cache.npz")):
            src_root = root
            break
    if src_root is None:
        found = [os.path.join(r, d) for r, ds, _ in os.walk("/kaggle/input") for d in ds]
        raise FileNotFoundError(
            f"RESEED_DATASET={RESEED_DATASET!r}: nothing under /kaggle/input holds ocr/*.mmd "
            f"or index/embed_cache.npz. Attach it to this kernel (Add Input) first. "
            f"Directories seen: {found[:40]}"
        )
    else:
        print(f"resuming from {src_root}")
        for sub in ("ocr", "interim", "index"):
            src = os.path.join(src_root, sub)
            dst = os.path.join("data", sub)
            if os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                shutil.copytree(src, dst, dirs_exist_ok=True)
                print(f"  copied {src} -> {dst} ({len(os.listdir(dst))} entries)")
else:
    print("fresh start -- no checkpoint to resume from")
"""))

    cells.append(md("""## 3. Materialize the full ~1046-page corpus (NOT the ANNOT subset)"""))
    cells.append(code("""!bash scripts/get_data.sh
!df -h /kaggle/working
"""))

    cells.append(
        md("""## 4. Write the code (region-routing Reader integration -- Step 28 point 11 --
isn't on `main` yet)""")
    )
    cells.append(
        code(
            'import os\n\nos.makedirs("src/doc_agent/vision", exist_ok=True)\nos.makedirs("KAGGLE/step30_full_ocr_reindex", exist_ok=True)\n'
        )
    )
    cells.append(code("%%writefile src/doc_agent/vision/ocr.py\n" + OCR_MODULE))
    cells.append(code("%%writefile configs/config.yaml\n" + CONFIG_YAML))
    cells.append(code("%%writefile KAGGLE/step30_full_ocr_reindex/run_step30.py\n" + RUN_STEP30))

    cells.append(md("""## 5. Smoke: tiny timing/correctness check before committing to the full run

plan.md Step 30's own explicit instruction — measure before committing, don't reuse
Step 16/18b's ~5h estimate (it was wrong: the real Step 18b run took ~13h51m)."""))
    cells.append(code("""!python KAGGLE/step30_full_ocr_reindex/run_step30.py --smoke
"""))

    if not INCLUDE_FULL_RUN:
        cells.append(md("""## (smoke-only push — stopping here)

This version does not run the full pipeline. Review the timing projection printed above,
then flip `INCLUDE_FULL_RUN = True` in `build_notebook.py`, regenerate, and push again for
the informed full run."""))
        return cells

    cells.append(md("""## 6. The real run

`--skip-known-failures` is set because this is a resume with the SAME reader. The resume
check is `mmd_path.exists()` and a failed page writes no `.mmd`, so without the flag all
53 recorded failures get re-run at full inference cost -- measured at 7561s (2.10h, 47.6%
of the whole OCR stage) on 2026-08-13 -- to reproduce the identical result, since decoding
is greedy and therefore deterministic. Skipped pages produce no chunks, exactly as
re-failing would, so the index is unchanged. **Drop this flag whenever the reader
changes**: a page one reader cannot read may well succeed under another, which is the
entire premise of Step 18b and Step 28."""))
    cells.append(code("""!df -h /kaggle/working
!python KAGGLE/step30_full_ocr_reindex/run_step30.py --skip-known-failures
!df -h /kaggle/working
"""))

    cells.append(md("""## 7. Package the output for download"""))
    cells.append(code("""out_dir = "/kaggle/working/out"
os.makedirs(out_dir, exist_ok=True)
for sub in ("data/ocr", "data/interim", "data/index"):
    if os.path.isdir(sub):
        shutil.copytree(sub, f"{out_dir}/{sub}", dirs_exist_ok=True)
archive_path = shutil.make_archive("/kaggle/working/step30_output", "zip", out_dir)
print(f"wrote {archive_path} ({os.path.getsize(archive_path) / 1e6:.1f} MB)")
"""))

    cells.append(md(f"""## Resuming after a timeout

If step 6 above didn't finish (Kaggle killed the session at the ~9h/12h ceiling):

1. **Download this version's Output** (`kaggle kernels output {OWNER}/{KERNEL_SLUG} -p ./out`,
   or the Output tab on kaggle.com) — `data/ocr/`, `data/interim/`, `data/index/` inside it
   survived even though the run didn't complete.
2. Upload it as a Kaggle Dataset: `kaggle datasets create -p ./out/data -u` the first time,
   or `kaggle datasets version -p ./out/data -m "resume after timeout"` after.
3. Attach that dataset to this kernel (Add Input), set `RESEED_DATASET` (cell 2 above) to
   its slug, set `FRESH_START = False`, and re-push: `kaggle kernels push -p KAGGLE/step30_full_ocr_reindex/`.
4. Cell 2's resume logic copies the checkpoint back in; `run_step30.py` reads
   `data/ocr/<page_id>.mmd` per page and skips whatever's already transcribed.

The OCR stage (data/ocr/*.mmd, one file per page) is the expensive part to redo — a
resumed push should only pay for pages that never finished, not the whole book again."""))

    return cells


def main() -> None:
    _check_embeds_complete()
    nb = {
        "cells": build_cells(),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = KAGGLE_DIR / "kaggle_step30_full_ocr_reindex.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(nb['cells'])} cells, {out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
