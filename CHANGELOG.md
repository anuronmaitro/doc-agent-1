# Changelog
## 0.1.0 — starter skeleton

### A2 progress ledger
> One line per merged step. This is how anyone resumes: last `[x]` = where we are,
> first `[ ]` = what to do next. Owners and step numbers follow `plan.md`.

- [x] Step 01 — repo bootstrap: deps, lockfiles, gitignore, green CI baseline — S1 (2105037) — 2026-08-08
- [x] Step 02 — configs/task.yaml + config.yaml — S2 (2105047)
- [x] Step 03 — data/provenance.md + scripts/get_data.sh — S3 (2105058)
- [x] Step 04 — notebooks/eda.ipynb — S1 (2105037)
- [x] Step 05 — grading_kit manifest + labels reconcile + 3 gold pages — S2
- [x] Step 06 — data/validate.py + data/versioning.py — S3
- [x] Step 07 — governance/pii.py (pipeline blocker) — S1
- [x] Step 08 — ingest/loader.py — S2
- [x] Step 09 — ingest/preprocess.py — S3
- [x] Step 10 — vision/layout.py — S1
- [x] Step 11 — vision/ocr.py (baseline reader) — S2
- [ ] Step 12 — eval/metrics.py — S3
- [ ] Step 13 — index/chunk.py (semantic chunking, bonus E4) — S1
- [ ] Step 14 — index/embed.py — S2
- [ ] Step 15 — index/store.py + scripts/build_index.sh — S3
- [ ] Step 16 — Kaggle: baseline full-book OCR + BEFORE number — S1
- [ ] Step 17 — annotation conventions + tooling — S2
- [ ] Step 18 — render the 164 annotation pages — S3
- [ ] Step 19 — annotate TEST batch A (18) — S1
- [ ] Step 20 — annotate TEST batch B (18) — S2
- [ ] Step 21 — annotate VALIDATION (20) — S3
- [ ] Step 22 — annotate TRAIN A (35) — S1
- [ ] Step 23 — annotate TRAIN B (35) — S2
- [ ] Step 24 — annotate TRAIN C (35) — S3
- [ ] Step 25 — NIST extraction (Stage A pairs) — S1
- [ ] Step 26 — synthetic degradation pipeline — S2
- [ ] Step 27 — training/ + LoRA adapt — S3
- [ ] Step 28 — Kaggle: fine-tune + learning curve — S1
- [ ] Step 29 — final TEST measurement (once) — S2
- [ ] Step 30 — Kaggle: re-OCR + rebuild index — S3
- [ ] Step 31 — notebooks/kb_demo.ipynb — S1
- [ ] Step 32 — reports/pipeline_diagram.md — S2
- [ ] Step 33 — configs/design_choices.md — S3
- [ ] Step 34 — A2 form sections 4 + 5 — S1
- [ ] Step 35 — A2 form sections 2 + 3 — S2
- [ ] Step 36 — A2 form sections 1 + 6 + 7 — S3
- [ ] Step 37 — transcripts x3 — all
- [ ] Step 38 — final checks + `a2-submit` tag — S1

### Step 04 notes (EDA — measured numbers now live in the notebook)
- `notebooks/eda.ipynb` runs top-to-bottom on the real corpus and is committed **with outputs**,
  so every figure the A2 form quotes is traceable to a visible cell (the grounding gate).
- Measured: **1050 printed + 32 front-matter = 1082 rendered pages**, 2298×3053 px @ 300 dpi
  grayscale, source PDF 78.6 MB, page images 0.94 GB, archive-OCR text layer 579,798 words
  (≈9.7× the 60k floor — reference only; the graded count must come from *our* OCR).
- The 29-chapter map is in the notebook; page counts per chapter sum to 1050.
- **Split is asserted, not asserted-in-prose:** build 20 ch / 712 pp · val 4 ch / 114 pp ·
  test 5 ch / 224 pp, with `assert`s that the three chapter sets are pairwise disjoint, that every
  chapter is assigned, and that all three A1 gold pages (243, 255, 360) land in *test*.
- **Corrected a claim that our own data contradicted.** A first draft flagged "18% faint pages"
  using an absolute cutoff (brightness>200 & contrast<40) inherited from A1 — but this corpus has
  mean brightness 195 / contrast 35, so that cutoff fires on ordinary pages. Replaced with a
  corpus-relative measure: brightness p5–p95 = 188–203, contrast p5–p95 = 26–43, and **1/60 (2%)**
  low-contrast outliers at >2σ. The narrow unimodal spread with no degraded tail is the actual
  evidence for classical-only preprocessing (`enhance.enabled: false`).
- Notebook kept to 361 KB by capping inline figure dpi and thumbnailing the three page images —
  the grader still sees real pages without a 1 MB file in git. `data/eda_summary.json` is gitignored.

### Step 01 notes (environment decisions worth knowing)
- **Python pinned to 3.12** (`.python-version`, `requires-python = ">=3.11,<3.13"`).
  The pinned `torch<2.4` and `faiss-cpu<1.9` publish **cp311/cp312 wheels only**, so 3.13+
  cannot install this lock at all. `uv` will fetch 3.12 automatically.
- **Added 3 dependencies:** `peft` (LoRA), `opencv-python-headless` (preprocessing/degradation),
  `pymupdf` (page rendering). Nougat needs no extra package — `transformers==4.44.2` supports it natively.
- **The starter did not pass its own CI gates.** Fixed with formatting + annotations only:
  `black` (73 files), `ruff --fix` (57 issues), `ruff ignore = F405/E501/N812` (star-import and
  black-overlap rules), and type annotations on stub signatures. **No function name, parameter name,
  contract field, pipeline order, seam, or tool name was changed** — `tests/test_structure.py`,
  `test_contracts.py` and `test_tools.py` all still pass.
- **`bandit` fix:** the only finding was B101 (`assert`) inside the FIXED `hooks.py`, so instead of
  editing that file we added `.bandit` (skips B101, documented) and pointed `security.yml` at it.
- **`cd / ship` fix:** the Dockerfile shipped `python:3.11-slim`, but `requirements.lock` is compiled
  under 3.12 and pins `numpy==2.5.1` (Requires-Python >=3.12) — the image could never install it.
  Base image is now `python:3.12-slim`, matching `.python-version` and `requires-python`. Also added
  the CPU-only torch index, since the default Linux wheel drags in ~2.5 GB of unused `nvidia-cuda-*`.
  torch/torchvision are installed first from the CPU-only `--index-url` so the source is deterministic
  (PEP 440 local-version rules mean `2.3.1+cpu` still satisfies the `torch==2.3.1` pin in the lock).
  The `cd` trigger fires on `workflow_dispatch` **and** on pushes touching `pyproject.toml`,
  `requirements.lock`, `Dockerfile`, or `cd.yml` — i.e. exactly when the image can break — so the
  "deps install on Linux + py3.12" check stays real without costing every unrelated commit.
  A4 TODO: implement the image push / HF Spaces deploy once `serve/api.py` answers.
