"""Training — unified entrypoint.

Orchestrates plan.md Step 28's "two-stage fine-tune" as two separate `Trainer.fit()` calls
against the SAME `LitComponent` instance (see lit_modules.py's class docstring for why one
instance, not two): Stage A (all of Step 25's NIST pairs, degraded on-the-fly, no early
stopping — volume/warmup) runs first, then Stage B (the 122 A&S train pages) continues
from Stage A's weights with a lower LR and early-stops on the 20 A&S val pages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from .datamodule import DocDataModule, make_collate_fn
from .lit_modules import LitComponent

logger = get_logger(__name__)


def _build_logger(cfg: dict[str, Any], run_name: str) -> Any:
    from lightning.pytorch.loggers import WandbLogger

    log_cfg = cfg.get("logging", {})
    # offline by default: a smoke/CI run must not require WANDB_API_KEY or network access
    # to pass. Step 28's real Kaggle run overrides wandb_mode to "online" once a key is
    # configured in that environment's secrets.
    return WandbLogger(
        project=log_cfg.get("wandb_project", "mathscholar-ocr-finetune"),
        name=run_name,
        mode=log_cfg.get("wandb_mode", "offline"),
    )


def _build_trainer(
    cfg: dict[str, Any], stage_cfg: dict[str, Any], *, early_stopping: bool, run_name: str
) -> L.Trainer:
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    callbacks: list[Any] = []
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=stage_cfg.get("early_stopping_monitor", "val_loss"),
                patience=int(stage_cfg.get("early_stopping_patience", 3)),
                mode="min",
                # strict=False: don't crash if val_loss hasn't been logged yet. Under real
                # training (full epochs) validation always runs before on_train_epoch_end,
                # so this never fires. It only matters for a smoke/debug run whose
                # max_steps cuts the first epoch short before any validation pass
                # completes -- found running the real smoke train, not assumed -- where
                # strict=True would otherwise crash a run that would have been fine at
                # real scale, for a reason that has nothing to do with the pipeline itself.
                strict=False,
            )
        )
    ckpt_dir = cfg.get("checkpoint", {}).get("dir")
    if ckpt_dir:
        callbacks.append(
            ModelCheckpoint(dirpath=Path(ckpt_dir) / run_name, save_top_k=1, monitor=None)
        )

    trainer_kwargs: dict[str, Any] = {
        "accelerator": "gpu" if str(cfg.get("device", "cpu")).startswith("cuda") else "cpu",
        "devices": 1,
        "max_epochs": int(stage_cfg.get("max_epochs", 1)),
        "logger": _build_logger(cfg, run_name),
        "callbacks": callbacks,
        "deterministic": True,
        "enable_progress_bar": True,
    }
    # Smoke-test / debug overrides: only applied when present in stage_cfg, so the real
    # Step 28 config (no such keys) is unaffected. Applied BEFORE the no-validation force
    # below, not after, so a future stage_a smoke override can never accidentally re-enable
    # a validation pass Stage A has no dataset for (see that block's own comment).
    for key in ("max_steps", "limit_train_batches", "limit_val_batches"):
        if key in stage_cfg:
            trainer_kwargs[key] = stage_cfg[key]

    if not early_stopping:
        # Stage A has no validation split (DocDataModule.val_dataloader() returns None --
        # plan.md Step 27's DECISION: Stage A is off-distribution volume/warmup, not the
        # model-selection signal). Lightning still runs an automatic pre-training "sanity
        # check" validation pass by default regardless of whether early stopping is
        # requested, which crashes on a None dataloader -- found running the real smoke
        # train, not assumed. Both settings below are required: limit_val_batches alone
        # still leaves the sanity check trying to iterate the None dataloader first. This
        # is intentionally the LAST thing set on trainer_kwargs (see above) so it cannot be
        # silently overridden.
        trainer_kwargs["limit_val_batches"] = 0
        trainer_kwargs["num_sanity_val_steps"] = 0

    return L.Trainer(**trainer_kwargs)


def main(component: str, cfg: dict) -> None:
    """Train one component with a seeded Lightning Trainer + W&B logger.

    `component` selects which sub-system to fine-tune; only "ocr" is implemented (plan.md
    Step 27's scope). Runs Stage A then Stage B in sequence — see module docstring.
    """
    L.seed_everything(int(cfg.get("seed", 42)), workers=True)

    lit = LitComponent(cfg, component=component)
    collate_fn = make_collate_fn(lit.processor, cfg["data"]["max_target_length"])

    logger.info("training.train: Stage A (NIST) starting")
    lit.set_stage(cfg["stage_a"])
    dm_a = DocDataModule(cfg, data_stage="nist", collate_fn=collate_fn)
    trainer_a = _build_trainer(cfg, cfg["stage_a"], early_stopping=False, run_name="stage_a_nist")
    trainer_a.fit(lit, datamodule=dm_a)
    logger.info("training.train: Stage A (NIST) complete")

    logger.info("training.train: Stage B (A&S) starting")
    lit.set_stage(cfg["stage_b"])
    dm_b = DocDataModule(cfg, data_stage="as", collate_fn=collate_fn)
    trainer_b = _build_trainer(cfg, cfg["stage_b"], early_stopping=True, run_name="stage_b_as")
    trainer_b.fit(lit, datamodule=dm_b)
    logger.info("training.train: Stage B (A&S) complete")
