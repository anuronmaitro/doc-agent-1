"""Step 27 — training smoke test.

Runs `doc_agent.training.train.main("ocr", cfg)` end-to-end against REAL data (Step 25's
NIST pairs, degraded on-the-fly; the real 122/20 A&S train/val pages) and the REAL pinned
facebook/nougat-base model + LoRA adapter, on CPU, for a handful of steps per stage --
proving the whole pipeline (datamodule -> collate -> LoRA-wrapped model -> loss -> optimizer
-> Lightning Trainer) actually runs without crashing, not just that it imports.

This is NOT the real fine-tune (that's Step 28, on a Kaggle GPU, with `configs/train_ocr.yaml`
UNMODIFIED). It overrides a handful of that config's keys purely for wall-clock speed:
`max_steps` per stage (10 + 10 = 20, plan.md Step 27's own "Done when" number),
`limit_val_batches`, a smaller `max_target_length` (long A&S page transcripts otherwise
dominate CPU forward/backward cost), and `wandb_mode: disabled` (no network dependency for
a smoke run). Everything else -- model, LoRA config, data sources, loss -- is identical to
the real config.

Usage:
    python scripts/smoke_train.py
    python scripts/smoke_train.py --steps 5   # even smaller, for a quick local check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doc_agent.training.train import main as train_main  # noqa: E402


def _load_cfg(path: str) -> dict:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", default="configs/train_ocr.yaml")
    p.add_argument("--steps", type=int, default=10, help="max_steps PER STAGE (default 10+10=20)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_cfg(args.cfg)

    cfg["logging"] = {**cfg.get("logging", {}), "wandb_mode": "disabled"}
    cfg["stage_a"] = {
        **cfg["stage_a"],
        "batch_size": 1,
        "max_steps": args.steps,
        "max_epochs": 1,
    }
    cfg["stage_b"] = {
        **cfg["stage_b"],
        "batch_size": 1,
        "max_steps": args.steps,
        "max_epochs": 1,
        "limit_val_batches": 2,
    }
    cfg["data"] = {**cfg["data"], "max_target_length": 256}
    # Never write into data/models/ocr_lora/ -- that path is committed (Step 28's real
    # adapter), and a smoke checkpoint left there would look like real training output.
    cfg["checkpoint"] = {"dir": "data/interim/smoke_train_ckpt"}

    t0 = time.time()
    train_main("ocr", cfg)
    elapsed = time.time() - t0
    print(f"smoke_train: {2 * args.steps} steps total ({args.steps}/stage) OK in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
