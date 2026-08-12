"""Step 28 — generates KAGGLE/kaggle.ipynb from scripts/run_finetune.py.

`KAGGLE/kaggle.ipynb` is a thin wrapper (clone main -> install -> materialize the Stage B
page images -> embed this script verbatim via `%%writefile` -> run it -> zip the output).
The embedded copy and the real file must never drift apart, so the notebook is generated
here rather than hand-edited on kaggle.com -- change `scripts/run_finetune.py`, then run:

    python scripts/build_kaggle_notebook.py

and push the regenerated `KAGGLE/kaggle.ipynb` (`kaggle kernels push -p KAGGLE/`).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = (REPO_ROOT / "scripts" / "run_finetune.py").read_text(encoding="utf-8")
DATAMODULE = (REPO_ROOT / "src" / "doc_agent" / "training" / "datamodule.py").read_text(
    encoding="utf-8"
)
TRAIN_MODULE = (REPO_ROOT / "src" / "doc_agent" / "training" / "train.py").read_text(
    encoding="utf-8"
)
OCR_MODULE = (REPO_ROOT / "src" / "doc_agent" / "vision" / "ocr.py").read_text(encoding="utf-8")
TRAIN_CFG = (REPO_ROOT / "configs" / "train_ocr.yaml").read_text(encoding="utf-8")

# Every one of these has, at least once, been a real file this notebook needed embedded
# (not cloned from `main`, since it wasn't committed there yet) and WASN'T -- run_finetune.py
# itself, then datamodule.py/train.py (SeedByEpochCallback), then vision/ocr.py (the
# repetition-detector fix) and configs/train_ocr.yaml (the epoch-count fix) at once, in the
# same push. `_check_embeds_complete` below is the guard against a fifth repeat: it fails
# loudly if any file this branch has touched under these paths isn't embedded somewhere in
# the generated notebook, rather than trusting this hardcoded list to stay complete by hand.
EMBEDDED_PATHS = (
    "scripts/run_finetune.py",
    "src/doc_agent/training/datamodule.py",
    "src/doc_agent/training/train.py",
    "src/doc_agent/vision/ocr.py",
    "configs/train_ocr.yaml",
)


def _check_embeds_complete(notebook_source_blob: str) -> None:
    """Fail loudly if `main` differs from the working tree anywhere under
    `src/doc_agent/`, `configs/train_ocr.yaml`, or `scripts/run_finetune.py`, for a path
    this notebook doesn't embed -- Kaggle clones `main` and would silently run the STALE
    version instead. Diffs against `main` (not just `git status`), so this also catches a
    file that's already committed on this branch but not yet merged."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "main",
            "--",
            "src/doc_agent",
            "configs/train_ocr.yaml",
            "scripts/run_finetune.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    changed = {ln.strip().replace("\\", "/") for ln in result.stdout.splitlines() if ln.strip()}
    # Untracked (brand new) files under the same paths -- `git diff` alone misses these.
    result_u = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "-uall",
            "--",
            "src/doc_agent",
            "configs/train_ocr.yaml",
            "scripts/run_finetune.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    for ln in result_u.stdout.splitlines():
        if ln.startswith("??"):
            changed.add(ln[3:].strip().replace("\\", "/"))

    missing = [p for p in changed if p not in EMBEDDED_PATHS]
    if missing:
        raise SystemExit(
            "build_kaggle_notebook: these files differ from main (or are untracked) and "
            "are NOT embedded in the notebook -- Kaggle would clone the stale `main` "
            "version instead:\n  "
            + "\n  ".join(sorted(missing))
            + "\n\nAdd them to EMBEDDED_PATHS and embed their content, or this push repeats "
            "the exact class of bug that already happened 4 times in Step 28."
        )
    # Also confirm every declared embed actually made it into the notebook, not just that
    # nothing extra is missing -- catches a typo in a %%writefile path silently.
    for path in EMBEDDED_PATHS:
        if f"%%writefile {path}" not in notebook_source_blob:
            raise SystemExit(f"build_kaggle_notebook: {path} is declared but never embedded")


OWNER = "eliasmainur"
KERNEL_SLUG = "mathscholar-step28-finetune"
REPO_URL = "https://github.com/anuronmaitro/doc-agent-1.git"


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

    cells.append(md("""# MathScholar (team 1) — Step 28: OCR fine-tune + learning curve

Owner: Elias Mainur (S3, 2105058), taking Step 28 in a swap with Anuron (S1), who took
Step 27 while Elias was offline — see `plan.md` Step 27's reassignment note and
`CHANGELOG.md`.

Runs `plan.md` Step 28: Step 27's two-stage LoRA fine-tune pipeline (Stage A = 695
degraded NIST pairs, Stage B = the 122 A&S train pages), at real GPU scale, across the
25 / 50 / 105 / 122-page learning curve, measured on the 20 A&S validation pages each time.

**Resumability across Kaggle's ~9h interactive / ~12h commit ceiling** (`plan.md` §11.4):
`scripts/run_finetune.py` writes `data/models/ocr_lora/run_state.json` after every
stage/curve-point boundary, not at the end — a re-run of this notebook reads that file
first and skips whatever it already marks done. See the **"Resuming after a timeout"**
section near the bottom before re-pushing this kernel.

**Do not edit this notebook directly on kaggle.com and expect it to survive** — it is
generated by `scripts/build_kaggle_notebook.py` from `scripts/run_finetune.py`. Change the
repo file and regenerate (`python scripts/build_kaggle_notebook.py`), so the two never
drift apart.
"""))

    cells.append(
        code(
            f"""FRESH_START = True  # False on a resumed push -- see "Resuming after a timeout" below
RESEED_DATASET = None  # e.g. "eliasmainur/mathscholar-step28-ckpt" -- set + FRESH_START=False to resume
REPO_URL = "{REPO_URL}"
BRANCH = "main"  # Step 27 is merged to main; Step 28's own new files are embedded below, not cloned
"""
        )
    )

    cells.append(md("""## 1. Clone the repo and install pinned dependencies

`main` already has everything Step 28 depends on: `src/doc_agent/training/*` (Step 27),
`configs/train_ocr.yaml`, `data/annot/nist/` (Step 25, committed), `data/annot/{train,val}/*.json`
(Steps 21-24, committed). It does **not** have `scripts/run_finetune.py` yet (that's this
step's own deliverable, still on Elias's local branch) or the train/val page **images**
(gitignored) -- both are handled in the cells below."""))

    cells.append(code("""import os
import subprocess

if not os.path.exists("/kaggle/working/repo"):
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, "/kaggle/working/repo"], check=True)
%cd /kaggle/working/repo
!git log --oneline -3
"""))

    cells.append(
        code(
            """# --no-cache-dir: pip's download cache otherwise sits on /kaggle/working's disk doing
# nothing useful after install -- one of several contributors to the disk exhaustion
# that crashed the first real run (see scripts/run_finetune.py's _no_ckpt_cfg docstring
# for the dominant cause, Lightning's own full-model checkpoint).
!pip install -q --no-cache-dir -r requirements.lock
# Sanity check for the exact bug plan.md Step 27 hit and pinned around (setuptools>=81
# drops pkg_resources, which `import lightning` needs) -- requirements.lock already pins
# setuptools==80.10.2, this just confirms the pinned install actually took.
import pkg_resources  # noqa: F401

print("pkg_resources OK")
!df -h /kaggle/working
"""
        )
    )

    cells.append(md("""## 2. Resume from a previous push's checkpoint (skip if this is a fresh run)

Only relevant after a timeout -- see **"Resuming after a timeout"** near the bottom. If
`RESEED_DATASET` is set and attached to this kernel as a data source (Kaggle mounts it at
`/kaggle/input/<dataset-slug>/`), this copies its `ocr_lora_ckpt/` (Stage A adapter +
`run_state.json` + any finished curve-point adapters) into the fresh clone's
`data/models/ocr_lora/` **before** training starts, so `run_finetune.py` sees the
already-done work and skips it."""))

    cells.append(code("""import shutil

if not FRESH_START and RESEED_DATASET:
    slug = RESEED_DATASET.split("/")[-1]
    src = f"/kaggle/input/{slug}/ocr_lora_ckpt"
    dst = "data/models/ocr_lora"
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"resumed: copied {src} -> {dst}")
        !ls -la data/models/ocr_lora
    else:
        print(f"WARNING: RESEED_DATASET set but {src} not found -- check the dataset is "
              f"attached (Add Input) and actually contains ocr_lora_ckpt/. Continuing FRESH.")
else:
    print("fresh start -- no checkpoint to resume from")
"""))

    cells.append(md("""## 3. Materialize the Stage B page images

`data/annot/train/*.png` and `data/annot/val/*.png` are gitignored (measured 96 MB, see
`plan.md` Step 18's `.gitignore` note) -- reproducible in one command instead. With
Internet On and no `data/pages/` present, this downloads the 78.6 MB A&S PDF (sha256-
checked) and renders only the 181 annotation pages directly from it -- not the full
1082-page corpus, so this is fast."""))

    cells.append(code("""!ANNOT=1 bash scripts/get_data.sh
"""))

    cells.append(md("""## 4. Write `scripts/run_finetune.py` and its dependencies

Five files embedded verbatim (see `EMBEDDED_PATHS` in `scripts/build_kaggle_notebook.py`
-- that script refuses to regenerate this notebook if any file under `src/doc_agent/`,
`configs/train_ocr.yaml`, or `scripts/run_finetune.py` differs from `main` and isn't in
this list, so this set of five is verified complete as of the last regeneration, not just
remembered by hand): `scripts/run_finetune.py` itself; `configs/train_ocr.yaml` (Stage A/B
epoch counts, raised from real evidence -- see plan.md Step 28); and three files owned by
earlier steps that needed real fixes found running this job --
`src/doc_agent/training/datamodule.py` (`SeedByEpochCallback` -- Stage A's degradation RNG
used to be seeded by pair index only, identical every epoch once `stage_a.max_epochs > 1`),
`src/doc_agent/training/train.py` (`_build_trainer` gained an `extra_callbacks` param),
and `src/doc_agent/vision/ocr.py` (`_failure_reason`'s repetition detector -- widened unit
cap + a new block-level duplicate check, found reading real predicted text from the first
completed run). **If you change any of the five real files, regenerate this notebook with
`scripts/build_kaggle_notebook.py`; do not hand-edit the embedded copies separately.**"""))

    cells.append(code("""import os

os.makedirs("src/doc_agent/training", exist_ok=True)
os.makedirs("src/doc_agent/vision", exist_ok=True)
"""))
    cells.append(code("%%writefile src/doc_agent/training/datamodule.py\n" + DATAMODULE))
    cells.append(code("%%writefile src/doc_agent/training/train.py\n" + TRAIN_MODULE))
    cells.append(code("%%writefile src/doc_agent/vision/ocr.py\n" + OCR_MODULE))
    cells.append(code("%%writefile configs/train_ocr.yaml\n" + TRAIN_CFG))
    cells.append(code("%%writefile scripts/run_finetune.py\n" + SCRIPT))

    cells.append(md("""## 5. Optional: Weights & Biases online logging

Off by default (`train.py`'s own rule, Step 27) -- this run works with `wandb_mode=offline`
and no key at all. If you want online logging, add a Kaggle **Secret** named
`WANDB_API_KEY` (Add-ons -> Secrets) before running this cell."""))

    cells.append(code("""try:
    from kaggle_secrets import UserSecretsClient
    key = UserSecretsClient().get_secret("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = key
    print("WANDB_API_KEY loaded from Kaggle Secrets -- wandb will log online")
except Exception:
    print("no WANDB_API_KEY secret found -- continuing with wandb offline (this is fine)")
"""))

    cells.append(md("""## 6. Smoke-check in the real environment

Local CPU smoke-testing of `run_finetune.py` hit an unrelated environment issue on the
dev machine (see the PR description) and couldn't be completed there -- this is the real
first execution of the new code, on the real target platform, before any GPU time is
spent on the actual curve. Tiny and CPU-fast even on a GPU kernel; writes its own state
file under `data/interim/smoke_run_finetune/`, never touching the real
`data/models/ocr_lora/` path."""))

    cells.append(code("""!python scripts/run_finetune.py --smoke
"""))

    cells.append(md("""## 7. Measure real GPU time before committing the full curve

`plan.md` Step 28's own instruction: don't assume a number, measure it -- the same
discipline Step 18b should have applied before its first full-book estimate (it didn't,
and the estimate was wrong by >2x). Prints a worst-case total and flags whether it risks
Kaggle's ~9h interactive ceiling."""))

    cells.append(code("""!python scripts/run_finetune.py --measure
"""))

    cells.append(md("""## 8. The real run

Resumable: safe to re-run this cell (or re-push this whole notebook) after an
interruption -- it reads `data/models/ocr_lora/run_state.json` first and continues from
whatever is already marked done, rather than restarting Stage A or a finished curve
point. Stage A trains once; each of the 4 curve points restarts from Stage A's saved
weights (see the script's own module docstring for why)."""))

    cells.append(code("""!df -h /kaggle/working
!python scripts/run_finetune.py
!df -h /kaggle/working
"""))

    cells.append(md("""## 9. Package the output for download

Two things go in the zip: `data/models/ocr_lora/` (the run_state + Stage A adapter + one
adapter directory per curve point -- what Step 28 commits to the repo) and
`reports/figures/step28_learning_curve.png` (the plot). Also copies just
`data/models/ocr_lora/` to a flat top-level `ocr_lora_ckpt/` -- if this run is later
attached as a Kaggle Dataset to reseed a resumed push (§2 above), that's the path this
notebook's own resume cell expects."""))

    cells.append(code("""import shutil

# shutil.make_archive (stdlib), not a shelled-out `zip` call -- guaranteed present
# regardless of what the Kaggle base image does or doesn't have installed.
os.makedirs("/kaggle/working/ocr_lora_ckpt", exist_ok=True)
shutil.copytree("data/models/ocr_lora", "/kaggle/working/ocr_lora_ckpt", dirs_exist_ok=True)

out_dir = "/kaggle/working/out"
os.makedirs(out_dir, exist_ok=True)
shutil.copytree("data/models/ocr_lora", f"{out_dir}/ocr_lora", dirs_exist_ok=True)
fig = "reports/figures/step28_learning_curve.png"
if os.path.exists(fig):
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    shutil.copy(fig, f"{out_dir}/figures/")
else:
    print("curve plot not present yet (not all curve points finished)")

archive_path = shutil.make_archive("/kaggle/working/step28_output", "zip", out_dir)
print(f"wrote {archive_path} ({os.path.getsize(archive_path) / 1e6:.1f} MB)")
print()
print("Download: notebook right sidebar -> Output -> step28_output.zip")
print("Unzip into the repo at data/models/ocr_lora/ and reports/figures/ -- see the PR")
print("description for the exact commit sequence (never committed from inside Kaggle).")
"""))

    cells.append(md("""## 10. Current curve status (read this before deciding to re-push)

Prints `run_state.json` -- which curve points are done, their validation numbers, and (if
incomplete) exactly what's left, so you know before spending more GPU hours whether this
finished or needs a resumed push."""))

    cells.append(code("""import json

state_path = "data/models/ocr_lora/run_state.json"
if os.path.exists(state_path):
    state = json.load(open(state_path))
    print("Stage A done:", state.get("stage_a_done"))
    for n in (25, 50, 105, 122):
        cp = state.get("curve_points", {}).get(str(n))
        if cp and cp.get("done"):
            m = cp["val_metrics"]
            print(f"  n={n:>3}: failure_rate={m['failure_rate']:.2f}  "
                  f"char_f1={m['mean_char_f1_among_successes']:.3f}  "
                  f"exact_match={m['formula_weighted_exact_match']:.3f}")
        else:
            print(f"  n={n:>3}: NOT DONE")
else:
    print("no run_state.json yet -- Stage A hasn't started or written its first checkpoint")
"""))

    cells.append(md(f"""## Resuming after a timeout

If step 8 above didn't finish (Kaggle killed the session at the ~9h/12h ceiling) before
you can reopen this:

1. **Try downloading this version's Output first** (`kaggle kernels output {OWNER}/{KERNEL_SLUG} -p ./out`,
   or Output tab on kaggle.com) -- if `ocr_lora_ckpt/run_state.json` is there, whatever
   finished survived even though the run didn't complete.
2. If it downloaded successfully, upload it as a Kaggle Dataset (or version an existing
   one): `kaggle datasets create -p ./out/ocr_lora_ckpt -u` the first time, or
   `kaggle datasets version -p ./out/ocr_lora_ckpt -m "resume after timeout"` after.
3. Attach that dataset to this kernel (Add Input), set `RESEED_DATASET` (cell 2 above) to
   its slug, set `FRESH_START = False`, and re-push:
   `kaggle kernels push -p KAGGLE/`.
4. Cell 2's resume logic copies the checkpoint back in; `run_finetune.py` reads
   `run_state.json` and continues from the next unfinished curve point.

If the download in step 1 comes back empty (Kaggle sometimes doesn't preserve
`/kaggle/working` for a run that errored/was killed outright, not just timed out
gracefully) there is nothing to resume from that push -- re-run from `FRESH_START = True`.
Stage A alone is the expensive shared step; everything after it is checkpointed per curve
point, so the worst case is redoing Stage A once, not the whole curve.
"""))

    return cells


def main() -> None:
    cells = build_cells()
    blob = "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    _check_embeds_complete(blob)

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = REPO_ROOT / "KAGGLE" / "kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(
        f"wrote {out} ({len(nb['cells'])} cells, {out.stat().st_size} bytes) "
        f"-- embed completeness verified against `main`"
    )


if __name__ == "__main__":
    main()
