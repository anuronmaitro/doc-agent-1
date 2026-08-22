"""Step 14 -- generates KAGGLE/step14_needs_research_probe/kaggle_step14_probe.ipynb.

One-off verification probe (plan_a3.md Step 14, Do item 5): before committing a
`needs_research` trigger/control pair to `grading_kit/tasks.jsonl`, empirically confirm
against the REAL production retrieval path -- `Retriever.retrieve()` + `is_weak()` +
`top_score()`, exactly as `agent.py`'s `decide()` calls them (Step 10) -- rather than assume
a query behaves a certain way. Clones `main` (already has retriever.py/config.py merged,
nothing new to embed) and mounts the real published index dataset, so this measures the
actual system, not a hand-rolled re-implementation of dense search.

Candidate queries are the honest byproduct of A2's own finding (`notebooks/kb_demo.ipynb`,
Step 31): numeric table-value lookups (dense LaTeX-packed tokens) rank badly against
BAAI/bge-m3, unlike prose+formula content. Several TEST-split table pages are tried as
trigger candidates, plus a couple of prose+formula control candidates, so a genuinely
verified pair can be picked from real results rather than gambling on one guess.

Regenerate with `python scripts/build_kaggle_notebook_step14_probe.py`, push with
`kaggle kernels push -p KAGGLE/step14_needs_research_probe/`.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OWNER = "himadribiswas0904"
KERNEL_SLUG = "mathscholar-step14-needs-research-probe"
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


PROBE_SCRIPT = """
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from doc_agent import config
from doc_agent.retrieval.retriever import Retriever, is_weak, top_score

cfg = config.load()
print(f"weak_threshold = {cfg['retrieve']['weak_threshold']}, k = {cfg['retrieve']['k']}")

retriever = Retriever(cfg)

# --- CONTROL candidates: expected to retrieve strongly at k=10, single pass. -------------
CONTROL_CANDIDATES = [
    {
        "id": "control_gamma_half",
        "query": "What is the exact value of the Gamma function at one-half, Gamma(1/2)?",
        "target_page": "as_p0255",
    },
    {
        # A2's own kb_demo.ipynb (Step 31) already confirmed this query's dense top-1 lands
        # on the right page/formula against this same index -- a second, independently
        # verified-good candidate rather than betting everything on one.
        "id": "control_exponential_integral",
        "query": "exponential integral Ei(x)",
        "target_page": "as_p0229",
    },
]

# --- TRIGGER candidates: dense numeric table-value lookups, the A2 kb_demo.ipynb finding's
# own pattern (packed LaTeX tokens, atypical for BAAI/bge-m3) -- several tried at once so a
# genuinely verified one can be picked from real results, not a single guess.
#
# ROUND 1 (kernel v3) result, kept here as a record: ALL FIVE round-1 candidates came back
# NOT weak (top_score 0.62-0.71, comfortably above weak_threshold=0.35) even though four of
# five had the WRONG top-1 chunk -- BAAI/bge-m3 is confidently wrong on these, not uncertain,
# so is_weak() would never fire and decide() would never widen. A real, reportable finding
# in itself, but none of them are valid needs_research:true candidates. Round 2 tries more
# obscure notation-heavy pages and terser, less-prose-like phrasing.
TRIGGER_CANDIDATES = [
    {
        "id": "trigger_elliptic_third_kind_table179",
        "query": "Pi(0.2; 75 degrees backslash 30 degrees) value from Table 17.9, elliptic integral of the third kind",
        "target_page": "as_p0625",
    },
    {
        "id": "trigger_jacobian_zeta_table177",
        "query": "Z(20 degrees backslash 10 degrees) value in the Jacobian zeta function table",
        "target_page": "as_p0619",
    },
    {
        "id": "trigger_aux_functions_table78",
        "query": "What is g(x) = g_2(u) at x^{-1} = 0.60 in the auxiliary functions table?",
        "target_page": "as_p0324",
    },
    {
        "id": "trigger_gamma_complex_table67_x14",
        "query": "At x=1.4, y=7.5, what is the imaginary part of ln Gamma(z) in the complex-argument gamma table?",
        "target_page": "as_p0281",
    },
    {
        "id": "trigger_expint_table55_col1",
        "query": "10^5 E_3(10) column (1) value in the exponential integrals large-argument table",
        "target_page": "as_p0234",
    },
    {
        # ROUND 2 also came back all-not-weak (top_score 0.62-0.74). One calibration check:
        # even a genuinely off-topic query ("boiling point of mercury") scored 0.523 -- still
        # above weak_threshold=0.35. ROUND 3 (final, per explicit agreement -- one more batch,
        # then stop regardless): dense-FORMULA pages instead of tables this time, all pages
        # that never appeared in the top-40 at all in reports/a3_retrieval_probe.md's own
        # header-query measurement (the worst-covered pages on record), phrased as natural
        # questions about genuinely obscure, deep-in-chapter identities.
        "id": "calibration_off_topic",
        "query": "What is the boiling point of mercury at standard atmospheric pressure?",
        "target_page": "NONE_EXPECTED_OUT_OF_CORPUS",
    },
    {
        "id": "trigger_bessel_other_diffeq",
        "query": "What differential equation does w = z^(1/2) times a Bessel-type function of lambda z satisfy?",
        "target_page": "as_p0362",
    },
    {
        "id": "trigger_erf_continued_fraction",
        "query": "What is the continued fraction expansion for 2 e^(z^2) times the integral from z to infinity of e^(-t^2) dt?",
        "target_page": "as_p0298",
    },
    {
        "id": "trigger_polygamma_integer_values",
        "query": "What is the formula for the nth polygamma function at integer argument z=1?",
        "target_page": "as_p0260",
    },
    {
        "id": "trigger_elliptic_circular_case_n_negative",
        "query": "How is the elliptic integral of the third kind reduced when n is negative, the circular case?",
        "target_page": "as_p0600",
    },
    {
        "id": "trigger_bessel_recurrence_relations",
        "query": "What is the recurrence relation between C_(nu-1) and C_(nu+1) for Bessel-type cylinder functions?",
        "target_page": "as_p0361",
    },
]

results = []
for label, candidates in [("control", CONTROL_CANDIDATES), ("trigger", TRIGGER_CANDIDATES)]:
    for cand in candidates:
        chunks = retriever.retrieve(cand["query"], k=cfg["retrieve"]["k"])
        weak = is_weak(chunks, cfg)
        score = top_score(chunks)
        top1_pages = chunks[0].page_ids if chunks else []
        top1_has_target = cand["target_page"] in top1_pages
        any_top10_has_target = any(cand["target_page"] in c.page_ids for c in chunks)
        row = {
            "label": label,
            "id": cand["id"],
            "query": cand["query"],
            "target_page": cand["target_page"],
            "top_score": score,
            "is_weak": weak,
            "top1_chunk_id": chunks[0].id if chunks else None,
            "top1_pages": top1_pages,
            "top1_has_target": top1_has_target,
            "target_anywhere_in_k10": any_top10_has_target,
        }
        results.append(row)
        print(
            f"[{label}] {cand['id']}: top_score={score:.4f} is_weak={weak} "
            f"top1_has_target={top1_has_target} target_in_k10={any_top10_has_target} "
            f"top1_pages={top1_pages}"
        )

out_path = Path("/kaggle/working/step14_needs_research_probe.json")
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\\nsaved {out_path}")

print("\\n=== summary: candidates matching the needed behaviour ===")
good_controls = [
    r for r in results if r["label"] == "control" and not r["is_weak"] and r["top1_has_target"]
]
good_triggers = [
    r for r in results if r["label"] == "trigger" and r["is_weak"]
]
print(f"controls that are strong (not weak) AND correct at top-1: {[r['id'] for r in good_controls]}")
print(f"triggers that are genuinely weak (is_weak=True):          {[r['id'] for r in good_triggers]}")
"""


def build_cells() -> list[dict]:
    cells: list[dict] = []

    cells.append(
        md("""# MathScholar (team 1) — Step 14: `needs_research` trigger/control verification probe

Owner: Himadri Gobinda Biswas (S2, 2105047).

**One-off verification, not a permanent eval artifact.** Before committing a
`needs_research` trigger/control pair to `grading_kit/tasks.jsonl` (Step 14,
`plan_a3.md`), this notebook checks several candidate queries against the REAL production
retrieval path — `Retriever.retrieve()` + `is_weak()` + `top_score()`, exactly what
`agent.py`'s `decide()` (Step 10) actually calls — over the real index (mounted from
`himadribiswas0904/mathscholar-index-mirror`, a byte-identical mirror of
`anuronmaitro/mathscholar-index` published because the original is private and not shared
with this account; file sizes verified to match `plan_a3.md`'s own Step 4 RESULT block
exactly before mirroring). A "trigger" that turns out answerable at k=10 would make the A3 agentic
gate see a single pass where it expected widening — fail-closed means that caps the grade,
so this is checked for real rather than assumed.

Candidates: one control (expected strong, single-pass) and five table-value-lookup trigger
candidates, all drawn from real `grading_kit/labels.jsonl` TEST-split pages — the same
*kind* of query A2's `notebooks/kb_demo.ipynb` (Step 31) found ranks badly (dense
LaTeX-packed table cells are atypical input for `BAAI/bge-m3`), tried against several real
table pages rather than gambling on the exact page A2 happened to test.

**Do not edit this notebook directly on kaggle.com** — regenerate it from
`scripts/build_kaggle_notebook_step14_probe.py` if the candidate queries change.
""")
    )

    cells.append(code(f"""REPO_URL = "{REPO_URL}"
BRANCH = "main"   # retriever.py / config.py / is_weak / top_score are all already merged
                   # (Steps 1, 10) -- nothing new needs embedding for this probe.
"""))

    cells.append(md("""## 1. Clone the repo and install only what this probe needs"""))

    cells.append(code("""import os
import subprocess

if not os.path.exists("/kaggle/working/repo"):
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, "/kaggle/working/repo"], check=True)
%cd /kaggle/working/repo
!git log --oneline -3
"""))

    cells.append(
        md("""## 2. Install dependencies -- deliberately NOT `pip install -r requirements.lock`

That installs the repo's OWN pins (`faiss-cpu>=1.8,<1.9`, `numpy<2` locally) -- Step 4's own
probe (`KAGGLE/a3_step04_index_probe/kaggle_step04.ipynb`) already hit this exact failure and
documented it: Kaggle's base image ships `numpy` 2.0.2, and installing the repo's old
`faiss-cpu` pin either downgrades `numpy` to satisfy it (breaking `scipy`, already imported
against numpy 2.x -- `ModuleNotFoundError: No module named 'numpy.char'`/`'numpy.strings'`)
or, with `--no-deps`, crashes on import anyway (an old `faiss-cpu` wheel compiled against
numpy 1.x cannot run under 2.0.2). Fix, proven working there: let pip pick whatever current
`faiss-cpu`/`sentence-transformers` it wants (no upper bound, both support numpy 2.x) and
otherwise rely on Kaggle's own pre-installed `torch`/`numpy`/`scipy`/`transformers` --
nothing needs downgrading in either direction. This notebook only reads an already-built
index back; it never touches the real index-build pipeline `pyproject.toml` governs, so a
looser constraint here is fine.""")
    )

    cells.append(code("""import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "faiss-cpu", "sentence-transformers", "pydantic"],
    check=True,
)
print("dependencies installed")
"""))

    cells.append(
        md("""## 3. Locate the mounted index dataset and put it where `index/store.py` expects it

`store.load()`'s `INDEX_DIR` is the relative path `data/index/` (cwd-relative, matching a
local `bash scripts/build_index.sh` run) — recursive search under `/kaggle/input`, same
"don't assume a mount depth" lesson Step 4's own probe already learned, then copied into
place rather than assumed to already be there.""")
    )

    cells.append(code("""import shutil
import zipfile
from pathlib import Path

print("locating mounted index dataset...")
faiss_files = sorted(Path("/kaggle/input").rglob("faiss.index"))
if not faiss_files:
    zips = sorted(Path("/kaggle/input").rglob("*.zip"))
    assert zips, "no faiss.index and no .zip found anywhere under /kaggle/input/"
    extract_dir = Path("/kaggle/working/index_extracted")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(extract_dir)
    faiss_files = sorted(extract_dir.rglob("faiss.index"))
assert faiss_files, "still no faiss.index found after checking for a zip"
mounted_index_dir = faiss_files[0].parent
print(f"mounted index dir: {mounted_index_dir}")

target_dir = Path("data/index")
target_dir.mkdir(parents=True, exist_ok=True)
for name in ("faiss.index", "chunks.jsonl", "index_meta.json"):
    src = mounted_index_dir / name
    if src.exists():
        shutil.copy(src, target_dir / name)
        print(f"copied {name}")
    else:
        print(f"WARNING: {name} not found in mounted dataset")
"""))

    cells.append(md("""## 4. Run the real retrieval path against the candidate queries

`Retriever.retrieve(query, k=10)` — the exact call `decide()` makes — then `is_weak()` /
`top_score()` on the result, exactly as `decide()` checks them. No hand-rolled dense search
here; this is the production code path, imported and called directly."""))

    cells.append(code(PROBE_SCRIPT))

    cells.append(md("""## 5. Package the output for download"""))

    cells.append(code("""import shutil as _shutil

out_dir = "/kaggle/working/out"
os.makedirs(out_dir, exist_ok=True)
_shutil.copy("/kaggle/working/step14_needs_research_probe.json", out_dir)

archive_path = _shutil.make_archive("/kaggle/working/step14_probe_output", "zip", out_dir)
print(f"wrote {archive_path} ({os.path.getsize(archive_path) / 1e6:.2f} MB)")
print("Download: notebook right sidebar -> Output -> step14_probe_output.zip")
"""))

    return cells


def main() -> None:
    cells = build_cells()
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_dir = REPO_ROOT / "KAGGLE" / "step14_needs_research_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "kaggle_step14_probe.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({len(nb['cells'])} cells, {out.stat().st_size} bytes)")

    kernel_metadata = {
        "id": f"{OWNER}/{KERNEL_SLUG}",
        "title": "MathScholar Step14 Needs Research Probe",
        "code_file": "kaggle_step14_probe.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,  # a handful of BGE-M3 query encodes -- CPU is plenty, and this
        # conserves the shared ~30 GPU-h/week quota for heavier steps
        # (Step 23 ablations, Step 26 RLVR training).
        "enable_internet": True,
        "dataset_sources": ["himadribiswas0904/mathscholar-index-mirror"],
        "competition_sources": [],
        "kernel_sources": [],
    }
    meta_out = out_dir / "kernel-metadata.json"
    meta_out.write_text(json.dumps(kernel_metadata, indent=1), encoding="utf-8")
    print(f"wrote {meta_out}")


if __name__ == "__main__":
    main()
