"""Stage C — table-focused continuation fine-tune, from the saved curve_n122 adapter.

Requested directly (not part of plan.md's original Step 25-30 sequence): Step 28's learning
curve showed the fine-tuned reader working well on formulas/prose but hallucinating on
dense numeric tables (`as_p0509`, `as_p0351` -- see plan.md Step 28's writeup). This script
continues fine-tuning FROM the best saved checkpoint (`data/models/ocr_lora/curve_n122`),
table-focused, rather than restarting Stage A/B from scratch.

Training data, two sources, both reused rather than newly risked:
- `data/annot/nist_tables/pairs.jsonl` (`scripts/extract_nist_tables.py`) -- 9 real NIST
  table pairs, degraded on-the-fly like Stage A. Deliberately small: most tables in that
  967-page book are complex multi-line "formula-pair" layouts that the extractor correctly
  refuses to guess at (same "skip rather than scramble" rule Step 25 used) rather than risk
  a silently wrong label -- see that script's own docstring for the measured breakdown.
- `AS_TABLE_DENSE_PAGES` below -- 26 of the 122 A&S train pages already identified as
  genuinely table-dense (>=3 numeric-row-like lines), reused verbatim from
  `data/annot/train/`. REAL, human-verified text, zero new extraction risk, whole-page like
  Stage B -- this is the primary volume here, since 9 NIST pairs alone is not "enough" for
  a meaningful continuation.

Catastrophic-forgetting guard -- the literal ask ("make sure our current model does not
degrade"), not just a training-loss number:
1. The starting checkpoint (`curve_n122`) is never overwritten. This script writes to
   `data/models/ocr_lora/table_ft/`, a new directory.
2. LR is well below Stage B's 2e-5 -- this is a small, delicate nudge on an already-
   converged model, not a fresh curriculum stage.
3. Early stopping monitors `val_loss` on the FULL 20 A&S validation pages, not a
   table-only subset -- a regression on non-table content stops training rather than being
   chased past by continued table-only gradient steps.
4. Before/after comparison reuses `run_finetune.py`'s own `_evaluate_on_val` on the SAME 20
   val pages, printed per-page (not just aggregate) so a hidden regression on any single
   page is visible.

Also runs the NON-training approach recommended alongside this continuation (plan.md
Step 28's table-weakness discussion, option 1): region-level generation on the two known
table-hallucination pages (`as_p0509`, `as_p0351`), using `vision.layout.detect()` to crop
just the table region and running `generate()` on that crop instead of the whole page --
an isolated crop is closer to the model's training distribution (Stage A/C's own crops
are formula/table-sized, not full dense two-column pages). Printed side by side with the
whole-page output for a human read, not scored formally (a region crop's natural output
scope differs from the whole-page gold text, so `ocr_f1` against full-page gold isn't a
fair number here) -- this is exploratory diagnostic evidence, not a pipeline change.

Usage (Kaggle, GPU, from the repo root):
    python scripts/run_finetune_tables.py
Local dry run (CPU, tiny, proves the control flow only):
    python scripts/run_finetune_tables.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from doc_agent.logging_conf import get_logger  # noqa: E402
from doc_agent.training.datamodule import make_collate_fn  # noqa: E402
from doc_agent.training.degrade import degrade_one  # noqa: E402
from doc_agent.training.lit_modules import LitComponent  # noqa: E402
from doc_agent.training.train import _build_trainer  # noqa: E402

from run_finetune import (  # noqa: E402
    _evaluate_on_val,
    _load_val_records,
    _no_ckpt_cfg,
    _save_adapter,
)

logger = get_logger(__name__)

# The 26 of 122 A&S train pages measured (>=3 numeric-row-like lines in their own
# human-corrected `text`) to be genuinely table-dense, not just a passing "Table" mention
# -- see the module docstring. `as_p0865`/`as_p0864` (13 rows each) and `as_p1017` (19
# rows) are the densest; all 26 are outside the 20-page VAL set (confirmed: `as_p0509` and
# `as_p0351`, the two pages that motivated this whole continuation, are VAL pages, so
# reusing them here would leak the held-out set -- they are correctly absent below).
AS_TABLE_DENSE_PAGES: tuple[str, ...] = (
    "as_p0024", "as_p0040", "as_p0044", "as_p0050", "as_p0051", "as_p0106", "as_p0114",
    "as_p0115", "as_p0127", "as_p0140", "as_p0141", "as_p0143", "as_p0161", "as_p0181",
    "as_p0748", "as_p0749", "as_p0765", "as_p0813", "as_p0864", "as_p0865", "as_p0915",
    "as_p0998", "as_p1000", "as_p1004", "as_p1010", "as_p1017",
)

# A second run tried diluting this 100% table-shaped mix with 20 non-table A&S pages
# (evenly spread across the book) plus a gentler LR (2.5e-6) and fewer epochs (3), on the
# theory that a 100%-table gradient was specializing away from ordinary formula/prose
# content. That theory was WRONG: the diluted run regressed MORE pages, not fewer
# (failure_rate 0.10->0.25 vs the kept run's 0.10->0.10), including two new failures
# (as_p0453, as_p0518) the kept run didn't have, and it recovered as_p0534 LESS well (still
# failed, vs the kept run's fix to f1=0.480). Whatever is driving the per-page regressions,
# it is not simply "too much table content, too high an LR, too many epochs" -- so this
# script keeps the configuration that produced the better (though still imperfect) result
# rather than the "gentler" one. See plan.md's Step 28 writeup / handoff notes for the full
# before/after comparison table.
NIST_TABLES_PAIRS_PATH = Path("data/annot/nist_tables/pairs.jsonl")
STARTING_ADAPTER_DIR = Path("data/models/ocr_lora/curve_n122")
OUT_DIR = Path("data/models/ocr_lora/table_ft")
STATE_PATH = OUT_DIR / "run_state.json"
LR = 5.0e-6  # well below Stage B's 2e-5 -- see module docstring point 2
MAX_EPOCHS = 6
EARLY_STOPPING_PATIENCE = 2  # tighter than Stage B's 3 -- this is a small delicate nudge,
                             # not a fresh curriculum; stop sooner if val_loss backslides


class _NistTableDataset:
    """Mirrors `training.datamodule._NistStageADataset` for the small NIST table set --
    not reused directly from there because it reads a different pairs.jsonl/out_dir and
    this project's own convention (see `extract_nist_pairs.py` / `extract_nist_tables.py`)
    keeps each extraction's own consumer close to its own script rather than growing one
    shared class's branching."""

    def __init__(self, pairs_path: Path, degradation_cfg: dict[str, Any], seed: int) -> None:
        self._pairs = [json.loads(ln) for ln in pairs_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not self._pairs:
            raise ValueError(f"_NistTableDataset: {pairs_path} is empty")
        self._deg_cfg = degradation_cfg
        self._seed = seed

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        rec = self._pairs[idx]
        src = Image.open(rec["image"]).convert("L")
        arr = np.asarray(src, dtype=np.float32)
        rng = np.random.default_rng(self._seed + idx)
        degraded = degrade_one(arr, self._deg_cfg, rng)
        image = Image.fromarray(degraded).convert("RGB")
        return {"image": image, "text": rec["text"]}


class _ASTableDataset:
    """The 26 table-dense A&S train pages, reused whole-page like Stage B -- real,
    human-corrected text, no degradation applied (these are real scans already, unlike
    the synthetic NIST crops)."""

    def __init__(self, page_ids: tuple[str, ...], train_dir: str) -> None:
        self._records: list[dict[str, Any]] = []
        for pid in page_ids:
            jp = Path(train_dir) / f"{pid}.json"
            row = json.loads(jp.read_text(encoding="utf-8"))
            png_path = jp.with_suffix(".png")
            if not png_path.exists():
                raise FileNotFoundError(
                    f"_ASTableDataset: {jp} has no sibling image {png_path} -- run "
                    "`ANNOT=1 bash scripts/get_data.sh` first"
                )
            self._records.append({"image_path": str(png_path), "text": row["text"]})

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from PIL import Image

        rec = self._records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        return {"image": image, "text": rec["text"]}


class _ConcatDataset:
    """Minimal concat -- avoids importing torch.utils.data.ConcatDataset just for this."""

    def __init__(self, a: Any, b: Any) -> None:
        self._a, self._b = a, b

    def __len__(self) -> int:
        return len(self._a) + len(self._b)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < len(self._a):
            return self._a[idx]
        return self._b[idx - len(self._a)]


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _lit_from_saved_adapter(cfg: dict[str, Any], adapter_dir: Path) -> LitComponent:
    """A LitComponent whose LoRA weights are the REAL saved curve_n122 adapter, not a
    freshly (randomly) initialized one. `LitComponent.__init__` always builds a fresh
    base + randomly-init LoRA wrap via `apply_lora` (Step 27's own construction path);
    this discards that fresh LoRA wrap and re-wraps the SAME underlying frozen base model
    with the real trained adapter loaded from disk (`PeftModel.from_pretrained`, the
    standard load path for a `save_pretrained` directory -- `_save_adapter` in
    run_finetune.py is what wrote it), rather than re-downloading the base model a second
    time. `is_trainable=True` is required -- `from_pretrained` defaults to False (it's
    built for inference-loading), which silently freezes every LoRA param and left
    `configure_optimizers()` with an empty parameter list (`ValueError: optimizer got an
    empty parameter list`) the first time this ran without it."""
    from peft import PeftModel

    lit = LitComponent(cfg, component="ocr")
    base = lit.model.get_base_model()
    lit.model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    return lit


KNOWN_TABLE_HALLUCINATION_PAGES: tuple[str, ...] = ("as_p0509", "as_p0351")


def _region_level_table_diagnostic(
    lit: LitComponent, device: str, val_annot_dir: str, max_new_tokens: int
) -> None:
    """Non-training comparison: whole-page vs. region-cropped generation, on the two
    pages that motivated this whole continuation. See module docstring for why this is
    printed for a human read rather than scored."""
    from PIL import Image as PILImage

    from doc_agent.contracts import Page
    from doc_agent.vision import layout
    from run_finetune import _generate_page

    print("\n=== Non-training diagnostic: whole-page vs. region-crop generation ===")
    for page_id in KNOWN_TABLE_HALLUCINATION_PAGES:
        jp = Path(val_annot_dir) / f"{page_id}.json"
        if not jp.exists():
            print(f"  {page_id}: not found under {val_annot_dir}, skipping")
            continue
        gold = json.loads(jp.read_text(encoding="utf-8"))["text"]
        png_path = jp.with_suffix(".png")
        full_image = PILImage.open(png_path).convert("RGB")

        whole_page_pred = _generate_page(lit, full_image, device, max_new_tokens)

        page = Page(id=page_id, image_path=str(png_path), doc_id="diagnostic")
        try:
            regions = layout.detect([page], {})
        except Exception as exc:  # layout.detect's real dependencies may not be present
            print(f"  {page_id}: layout.detect() failed ({type(exc).__name__}: {exc}), "
                  "skipping region-crop comparison for this page")
            continue
        table_regions = [r for r in regions if r.kind == "table"]
        if not table_regions:
            print(f"  {page_id}: no table region detected, skipping region-crop comparison")
            continue

        print(f"\n--- {page_id} ---")
        print(f"GOLD (first 300 chars): {gold[:300]}")
        print(f"WHOLE-PAGE pred (first 300 chars): {whole_page_pred[:300]}")
        for i, region in enumerate(table_regions):
            crop = full_image.crop(region.bbox)
            if crop.width < 8 or crop.height < 8:
                continue
            region_pred = _generate_page(lit, crop, device, max_new_tokens)
            print(f"REGION[{i}] bbox={region.bbox} pred (first 300 chars): {region_pred[:300]}")


def main() -> None:
    # _build_trainer sets deterministic=True (same as Stage A/B); Lightning turns that into
    # torch.use_deterministic_algorithms(True), which on this session's cuBLAS build refused
    # to run without this workspace config and raised RuntimeError at the first LoRA forward
    # pass. Must be set before the first CUDA matmul, so as early as possible.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", default="configs/train_ocr.yaml")
    p.add_argument("--smoke", action="store_true", help="tiny CPU dry run of the control flow")
    args = p.parse_args()

    cfg = _load_yaml(args.cfg)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["device"] = device
    cfg["logging"] = {
        **cfg.get("logging", {}),
        "wandb_mode": "online" if os.environ.get("WANDB_API_KEY") else "disabled",
    }

    max_epochs = MAX_EPOCHS
    nist_pairs_path = NIST_TABLES_PAIRS_PATH
    ast_page_ids = AS_TABLE_DENSE_PAGES
    starting_adapter = STARTING_ADAPTER_DIR
    out_dir = OUT_DIR
    state_path = STATE_PATH
    if args.smoke:
        max_epochs = 1
        ast_page_ids = ast_page_ids[:2]
        out_dir = Path("data/interim/smoke_table_ft")
        state_path = out_dir / "run_state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("done"):
            logger.info("run_finetune_tables: already done (resumed) -- nothing to do")
            return
    else:
        state = {"done": False}

    logger.info(f"run_finetune_tables: loading starting adapter from {starting_adapter}")
    lit = _lit_from_saved_adapter(cfg, starting_adapter)
    if device.startswith("cuda"):
        lit = lit.to(device)

    collate_fn = make_collate_fn(lit.processor, cfg["data"]["max_target_length"])
    val_records = _load_val_records(cfg["data"]["val_annot_dir"])
    if args.smoke:
        val_records = val_records[:2]

    logger.info("run_finetune_tables: evaluating BEFORE continuation (baseline = curve_n122)")
    t0 = time.time()
    before_metrics = _evaluate_on_val(lit, val_records, device, cfg["data"]["max_target_length"])
    logger.info(
        f"run_finetune_tables: BEFORE -- failure_rate={before_metrics['failure_rate']:.2f} "
        f"char_f1={before_metrics['mean_char_f1_among_successes']:.3f} "
        f"({time.time() - t0:.0f}s)"
    )
    state["before_metrics"] = before_metrics
    _atomic_write_json(state_path, state)

    degradation_cfg = _load_yaml(cfg["data"]["degradation_cfg"])
    nist_ds = _NistTableDataset(nist_pairs_path, degradation_cfg, cfg["seed"])
    ast_ds = _ASTableDataset(ast_page_ids, cfg["data"]["train_annot_dir"])
    train_ds = _ConcatDataset(nist_ds, ast_ds)
    logger.info(f"run_finetune_tables: table training set = {len(nist_ds)} NIST + "
                f"{len(ast_ds)} A&S = {len(train_ds)} pairs")

    import lightning as L

    class _TableDataModule(L.LightningDataModule):
        """val comes from the SAME 20 A&S val pages every other stage uses
        (catastrophic-forgetting guard, see module docstring point 3), not a table-only
        subset. Must subclass L.LightningDataModule (not a bare protocol class) --
        Trainer.fit()'s internal is_overridden() hook check requires a real Lightning
        parent class and raises ValueError before any training step otherwise."""

        def __init__(self) -> None:
            super().__init__()
            self.train_dataset = train_ds

        def train_dataloader(self):  # noqa: ANN201
            from torch.utils.data import DataLoader

            return DataLoader(
                train_ds, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn,
            )

        def val_dataloader(self):  # noqa: ANN201
            from torch.utils.data import DataLoader

            class _ValWrap:
                def __len__(self_inner) -> int:
                    return len(val_records)

                def __getitem__(self_inner, idx: int) -> dict[str, Any]:
                    from PIL import Image

                    rec = val_records[idx]
                    image = Image.open(rec["image_path"]).convert("RGB")
                    return {"image": image, "text": rec["text"]}

            return DataLoader(
                _ValWrap(), batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn,
            )

    lit.set_stage({"lr": LR})
    stage_cfg = {
        "lr": LR, "max_epochs": max_epochs,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_monitor": "val_loss",
    }
    if args.smoke:
        stage_cfg["max_steps"] = 3
        stage_cfg["limit_val_batches"] = 1
    trainer = _build_trainer(
        _no_ckpt_cfg(cfg), stage_cfg, early_stopping=True, run_name="stage_c_tables",
    )
    t0 = time.time()
    trainer.fit(lit, datamodule=_TableDataModule())
    train_elapsed = time.time() - t0

    if device.startswith("cuda"):
        lit = lit.to(device)  # Trainer.fit's own teardown can move it back to CPU

    logger.info("run_finetune_tables: evaluating AFTER continuation")
    t0 = time.time()
    after_metrics = _evaluate_on_val(lit, val_records, device, cfg["data"]["max_target_length"])
    eval_elapsed = time.time() - t0
    logger.info(
        f"run_finetune_tables: AFTER -- failure_rate={after_metrics['failure_rate']:.2f} "
        f"char_f1={after_metrics['mean_char_f1_among_successes']:.3f} "
        f"(train {train_elapsed:.0f}s, eval {eval_elapsed:.0f}s)"
    )

    _save_adapter(lit, out_dir)

    if not args.smoke:
        _region_level_table_diagnostic(
            lit, device, cfg["data"]["val_annot_dir"], cfg["data"]["max_target_length"]
        )

    # Per-page before/after diff -- the real regression check, not just aggregate deltas.
    before_by_page = {r["page_id"]: r for r in before_metrics["per_page"]}
    after_by_page = {r["page_id"]: r for r in after_metrics["per_page"]}
    regressions = []
    for pid, after_row in after_by_page.items():
        before_row = before_by_page.get(pid, {})
        before_f1 = before_row.get("char_f1", 0.0 if before_row.get("failed") else None)
        after_f1 = after_row.get("char_f1", 0.0 if after_row.get("failed") else None)
        if before_f1 is not None and after_f1 is not None and after_f1 < before_f1 - 0.05:
            regressions.append({"page_id": pid, "before_f1": before_f1, "after_f1": after_f1})

    state.update({
        "done": True,
        "train_pairs": {"nist": len(nist_ds), "as_table_dense": len(ast_ds)},
        "train_elapsed_s": train_elapsed,
        "eval_elapsed_s": eval_elapsed,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "regressions_over_0.05_f1": regressions,
        "adapter_dir": str(out_dir),
    })
    _atomic_write_json(state_path, state)

    print("\n=== Stage C (table-focused continuation) summary ===")
    print(f"BEFORE: failure_rate={before_metrics['failure_rate']:.2f}  "
          f"char_f1={before_metrics['mean_char_f1_among_successes']:.3f}")
    print(f"AFTER:  failure_rate={after_metrics['failure_rate']:.2f}  "
          f"char_f1={after_metrics['mean_char_f1_among_successes']:.3f}")
    if regressions:
        print(f"\n⚠️  {len(regressions)} page(s) regressed by >0.05 char-F1:")
        for r in regressions:
            print(f"  {r['page_id']}: {r['before_f1']:.3f} -> {r['after_f1']:.3f}")
    else:
        print("\nNo page regressed by more than 0.05 char-F1.")


if __name__ == "__main__":
    main()
