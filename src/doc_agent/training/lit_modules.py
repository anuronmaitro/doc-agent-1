"""Training — Lightning modules per trainable component"""

from __future__ import annotations

from typing import Any

import lightning as L
import torch

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from .adapt import apply_lora

logger = get_logger(__name__)


class LitComponent(L.LightningModule):
    """Wraps the OCR reader (facebook/nougat-base, LoRA-adapted) for fine-tuning.

    One instance is reused across BOTH training stages (plan.md Step 28's "two-stage
    fine-tune"): `train.py` calls `set_stage("stage_a")` then `Trainer.fit()`, then
    `set_stage("stage_b")` and `Trainer.fit()` again on the SAME `LitComponent` — so the
    model's weights (and LoRA adapter state) carry over from Stage A into Stage B, while
    `configure_optimizers()` picks up each stage's own LR fresh (a new `Trainer.fit()` call
    always re-invokes `configure_optimizers()`, per Lightning's own documented behavior).

    Only "ocr" is implemented — the only component plan.md Step 27 asks for.
    """

    def __init__(self, cfg: dict[str, Any], component: str = "ocr") -> None:
        super().__init__()
        if component != "ocr":
            raise NotImplementedError(
                f"LitComponent: component={component!r} not implemented — "
                "Step 27 built the OCR reader fine-tune only (plan.md's Do-list)"
            )
        self.cfg = cfg
        self.stage_cfg: dict[str, Any] = cfg["stage_a"]  # default; train.py sets per-phase
        self._build_model()

    def _build_model(self) -> None:
        from transformers import NougatProcessor, VisionEncoderDecoderModel

        ocr_cfg = self.cfg["ocr"]
        model_name = ocr_cfg.get("model", "facebook/nougat-base")
        # Same pinned-revision rationale as vision/ocr.py's NOUGAT_REVISION (bandit B615:
        # an unpinned model name can resolve to different weights later).
        revision = ocr_cfg.get("revision") if model_name == "facebook/nougat-base" else None

        self.processor = NougatProcessor.from_pretrained(model_name, revision=revision)
        base_model = VisionEncoderDecoderModel.from_pretrained(model_name, revision=revision)
        # The nougat-base checkpoint's top-level VisionEncoderDecoderConfig ships with
        # decoder_start_token_id/pad_token_id both None -- fine for generate() (Nougat's
        # own generation_config path resolves the start token a different way), but
        # forward(labels=...) calls shift_tokens_right(), which reads self.config.
        # decoder_start_token_id directly and raises ValueError if it's None. Confirmed by
        # loading the real checkpoint and inspecting it, not assumed: the decoder's own
        # sub-config already has bos_token_id=0, matching the tokenizer's bos_token_id, so
        # that's the correct value to set, not a guess.
        base_model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        base_model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model = apply_lora(base_model, self.cfg["lora"])

    def set_stage(self, stage_cfg: dict[str, Any]) -> None:
        """Switch the active stage's hyperparameters (LR etc.) before the next
        `Trainer.fit()` call — see class docstring for why this is a mutation on the same
        instance rather than constructing a new LitComponent per stage."""
        self.stage_cfg = stage_cfg

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor | None = None) -> Any:
        return self.model(pixel_values=pixel_values, labels=labels)

    def training_step(self, batch: Any, idx: int) -> Any:
        outputs = self(batch["pixel_values"], labels=batch["labels"])
        self.log("train_loss", outputs.loss, prog_bar=True, on_step=True, on_epoch=True)
        return outputs.loss

    def validation_step(self, batch: Any, idx: int) -> Any:
        outputs = self(batch["pixel_values"], labels=batch["labels"])
        self.log("val_loss", outputs.loss, prog_bar=True, on_epoch=True)
        return outputs.loss

    def configure_optimizers(self) -> Any:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable,
            lr=float(self.stage_cfg["lr"]),
            weight_decay=float(self.cfg.get("optimizer", {}).get("weight_decay", 0.01)),
        )
