# Changelog
## 0.1.0 — starter skeleton

### A2 progress ledger
> One line per merged step. This is how anyone resumes: last `[x]` = where we are,
> first `[ ]` = what to do next. Owners and step numbers follow `plan.md`.

- [x] Step 01 — repo bootstrap: deps, lockfiles, gitignore, green CI baseline — S1 (2105037) — 2026-08-08
- [ ] Step 02 — configs/task.yaml + config.yaml — S2 (2105047)
- [ ] Step 03 — data/provenance.md + scripts/get_data.sh — S3 (2105058)
- [ ] Step 04 — notebooks/eda.ipynb — S1
- [ ] Step 05 — grading_kit manifest + labels reconcile + 3 gold pages — S2
- [ ] Step 06 — data/validate.py + data/versioning.py — S3
- [ ] Step 07 — governance/pii.py (pipeline blocker) — S1
- [ ] Step 08 — ingest/loader.py — S2
- [ ] Step 09 — ingest/preprocess.py — S3
- [ ] Step 10 — vision/layout.py — S1
- [ ] Step 11 — vision/ocr.py (baseline reader) — S2
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
