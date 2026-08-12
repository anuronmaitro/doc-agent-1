"""Training — Lightning datamodule.

Two data stages, matching plan.md Step 28's "two-stage fine-tune" (Stage A -> Stage B, not
one mixed dataset — see `train.py`, which runs `Trainer.fit()` twice against two separate
`DocDataModule` instances built from this file, continuing the SAME `LitComponent` weights
across both calls):

- **Stage A ("nist")** — Step 25's 695 NIST formula crops, degraded ON THE FLY per sample
  (see `_NistStageADataset`) using the exact same ops `scripts/degrade_nist_pairs.py` uses
  for its committed report figures (`doc_agent.training.degrade`, extracted from that
  script at this step so the two never drift apart). No pre-materialized
  `data/interim/nist_degraded/` directory is required or read — training never depends on
  that directory existing, only on Step 25's `data/annot/nist/pairs.jsonl` +
  `images/*.png`, both committed. No validation split (see plan.md Step 27's DECISION:
  Stage A is off-distribution volume/warmup, not the model-selection signal).
- **Stage B ("as")** — the 122 A&S hand-annotated train pages + 20 val pages
  (`data/annot/train|val/*.json` + their sibling `.png`). The chapter-disjoint split is
  already encoded by which directory a page's JSON lives in (Steps 18b-24's
  `doc_agent.data.validate.ANNOT_SETS`); `setup()` re-asserts each loaded page's chapter
  against that lock rather than re-deriving the split, so a corrupted/misplaced file fails
  loudly here instead of silently leaking chapters between train and val.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import lightning as L
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from .degrade import degrade_one

logger = get_logger(__name__)


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class _NistStageADataset(Dataset):
    """Stage A: one NIST formula crop per item, degraded on-the-fly (never from a
    pre-materialized directory) so training has no dependency on
    `scripts/degrade_nist_pairs.py` having been run first."""

    def __init__(self, pairs_path: str, degradation_cfg: dict[str, Any], seed: int) -> None:
        self._pairs = _load_jsonl(pairs_path)
        if not self._pairs:
            raise ValueError(f"_NistStageADataset: {pairs_path} is empty")
        self._deg_cfg = degradation_cfg
        self._seed = seed
        # Bumped by `SeedByEpochCallback` before each epoch (default 0, i.e. unchanged
        # behavior for a 1-epoch run). See that callback's docstring for why this exists:
        # without it, every epoch beyond the first would replay the IDENTICAL degraded
        # image per pair rather than a new augmented variant.
        self.epoch = 0

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self._pairs[idx]
        src = Image.open(rec["image"]).convert("L")
        arr = np.asarray(src, dtype=np.float32)
        # Per-sample-seeded RNG (config seed + pair index + epoch * dataset size) -- any
        # given (index, epoch) pair's degraded image is reproducible across runs/workers
        # without needing to persist it. The epoch term was added at Step 28 (found running
        # the real multi-epoch Stage A on Kaggle, see SeedByEpochCallback): without it this
        # was `seed + idx` alone, identical every epoch regardless of how many epochs ran --
        # fine for the 1-epoch default this shipped with, silently wrong for >1.
        rng = np.random.default_rng(self._seed + idx + self.epoch * len(self._pairs))
        degraded = degrade_one(arr, self._deg_cfg, rng)
        image = Image.fromarray(degraded).convert("RGB")
        return {"image": image, "text": rec["text"]}


class SeedByEpochCallback(L.Callback):
    """Advances `_NistStageADataset.epoch` before each training epoch, so a multi-epoch
    Stage A run degrades each pair differently per epoch instead of replaying the same
    image (see that dataset's own comment for the bug this fixes -- found running the real
    Step 28 Kaggle GPU job with `stage_a.max_epochs > 1`). No-op for a 1-epoch run (Step
    27's own smoke test and default config), since `on_train_epoch_start` only ever sets
    `epoch = 0` there -- safe to always attach, not something that needs conditioning on
    `max_epochs`."""

    def __init__(self, dataset: _NistStageADataset) -> None:
        self._dataset = dataset

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:  # noqa: ARG002
        self._dataset.epoch = trainer.current_epoch


class _ASStageBDataset(Dataset):
    """Stage B: one A&S annotated page per item -- the hand-corrected `text` (full page,
    real backslash-LaTeX) paired with that page's own 300dpi grayscale render."""

    def __init__(
        self,
        annot_dir: str,
        expected_chapters: frozenset[str] | None = None,
        max_pages: int | None = None,
    ) -> None:
        from ..data.validate import ANNOT_EXPECTED_COUNTS
        from ..ingest.loader import _chapter_of

        split = Path(annot_dir).name  # "train" or "val"
        json_paths = sorted(Path(annot_dir).glob("*.json"))
        if not json_paths:
            raise FileNotFoundError(
                f"_ASStageBDataset: no *.json under {annot_dir} -- run "
                "`ANNOT=1 bash scripts/get_data.sh` first if this is a fresh clone "
                "(images are gitignored; see .gitignore's data/annot/val|train comment)"
            )
        expected = ANNOT_EXPECTED_COUNTS.get(split)
        if max_pages is None and expected is not None and len(json_paths) != expected:
            raise ValueError(
                f"_ASStageBDataset: {annot_dir} has {len(json_paths)} pages, expected "
                f"{expected} (doc_agent.data.validate.ANNOT_EXPECTED_COUNTS[{split!r}]) -- "
                "a missing or extra file here silently leaks/shrinks the fine-tune's "
                "train/val split, so this is a hard error, not a warning"
            )
        if max_pages is not None:
            # Step 28's learning curve (25/50/105/122 train pages): a fixed sort order
            # (page_id, already the glob's sort key) makes each smaller curve point a
            # PREFIX of every larger one -- 25 pages ⊂ 50 ⊂ 105 ⊂ 122 -- rather than an
            # independently-resampled subset, so successive curve points differ only by
            # which pages were ADDED, which is what makes "did going from 105->122 help"
            # (plan.md Step 28 point 2) a meaningful comparison instead of confounded by
            # also swapping out which pages were included. Never applied to val (the
            # 20-page val set stays whole and identical across every curve point, which is
            # what makes the curve's y-axis comparable point to point).
            json_paths = json_paths[:max_pages]

        self._records: list[dict[str, Any]] = []
        for jp in json_paths:
            row = json.loads(jp.read_text(encoding="utf-8"))
            png_path = jp.with_suffix(".png")
            if not png_path.exists():
                raise FileNotFoundError(
                    f"_ASStageBDataset: {jp} has no sibling image {png_path} -- "
                    "run `ANNOT=1 bash scripts/get_data.sh` to materialize it "
                    "(gitignored, reproducible, byte-identical to data/pages/)"
                )
            if expected_chapters is not None:
                actual = _chapter_of(row["printed_page"])
                if actual not in expected_chapters:
                    raise ValueError(
                        f"_ASStageBDataset: LEAK — {row['page_id']} (chapter {actual}) is "
                        f"in {annot_dir} but that chapter is not one of {split}'s allowed "
                        f"chapters. See doc_agent.data.validate.ANNOT_SETS."
                    )
            if not row.get("text", "").strip():
                # A page whose annotation JSON exists but was never filled in (text=="")
                # is an in-progress annotation, not a training sample -- silently training
                # on an empty target would just teach the model to predict nothing.
                raise ValueError(
                    f"_ASStageBDataset: {jp} has empty text -- annotation incomplete, "
                    "not ready to train on"
                )
            self._records.append({"image_path": str(png_path), "text": row["text"]})

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self._records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        return {"image": image, "text": rec["text"]}


def make_collate_fn(processor: Any, max_target_length: int) -> Any:
    """Builds a collate_fn closing over a loaded NougatProcessor -- kept as a factory
    (not a bare module-level function) since the processor must be loaded once by
    `LitComponent` and shared with the datamodule, not reloaded per batch."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]
        pixel_values = processor(images, return_tensors="pt").pixel_values
        enc = processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_length,
        )
        labels = enc.input_ids.clone()
        labels[enc.attention_mask == 0] = -100  # ignore padding in the loss
        return {"pixel_values": pixel_values, "labels": labels}

    return collate


class DocDataModule(L.LightningDataModule):
    """`data_stage` selects Stage A ("nist") or Stage B ("as") -- see module docstring for
    why these are two separate DocDataModule instances rather than one mixed dataset.
    `collate_fn` is injected (not built internally) so it shares `LitComponent`'s already-
    loaded processor instead of this datamodule loading a second copy of it."""

    def __init__(
        self,
        cfg: dict[str, Any],
        data_stage: Literal["nist", "as"],
        collate_fn: Any,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data_stage = data_stage
        self.collate_fn = collate_fn
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        data_cfg = self.cfg["data"]
        if self.data_stage == "nist":
            degradation_cfg = _load_yaml(data_cfg["degradation_cfg"])
            self.train_dataset = _NistStageADataset(
                data_cfg["nist_pairs_path"], degradation_cfg, self.cfg["seed"]
            )
            self.val_dataset = None
            logger.info(f"DocDataModule[nist]: {len(self.train_dataset)} Stage A crops")
        elif self.data_stage == "as":
            from ..data.validate import BUILD_CHAPTERS, VAL_CHAPTERS

            # Step 28's learning curve (25/50/105/122 train pages): set once, here, via
            # cfg["data"]["stage_b_max_train_pages"] -- never applied to val, so the same
            # 20 pages are used at every curve point (see _ASStageBDataset's own comment
            # on why that's what keeps the curve's points comparable).
            max_train_pages = data_cfg.get("stage_b_max_train_pages")
            self.train_dataset = _ASStageBDataset(
                data_cfg["train_annot_dir"], BUILD_CHAPTERS, max_pages=max_train_pages
            )
            self.val_dataset = _ASStageBDataset(data_cfg["val_annot_dir"], VAL_CHAPTERS)
            logger.info(
                f"DocDataModule[as]: {len(self.train_dataset)} train / "
                f"{len(self.val_dataset)} val pages"
            )
        else:
            raise ValueError(f"DocDataModule: unknown data_stage={self.data_stage!r}")

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DocDataModule.train_dataloader called before setup()")
        stage_cfg = self.cfg["stage_a"] if self.data_stage == "nist" else self.cfg["stage_b"]
        return DataLoader(
            self.train_dataset,
            batch_size=int(stage_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(stage_cfg.get("num_workers", 0)),
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        stage_cfg = self.cfg["stage_b"]
        return DataLoader(
            self.val_dataset,
            batch_size=int(stage_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(stage_cfg.get("num_workers", 0)),
            collate_fn=self.collate_fn,
        )
