"""Step 28 — the actual Kaggle GPU fine-tune + learning curve.

Step 27 built the pipeline (`doc_agent.training.{datamodule,lit_modules,adapt,train}`) and
proved it runs end-to-end on CPU with `smoke_train.py`. This script RUNS that pipeline at
real scale on a Kaggle GPU: it does not add new training logic, it orchestrates Step 27's
existing pieces the way plan.md Step 28 asks for --- Stage A once, then Stage B four times
(25 / 50 / 105 / 122 A&S train pages), each measured on the same 20 validation pages.

Why this is a separate script and not four calls to `training.train.main()`:
`train.main()` runs Stage A immediately followed by Stage B on ONE `LitComponent`, which is
exactly right for a single run but wrong for a learning curve --- calling it four times would
retrain Stage A four times (four passes over the same 695 NIST pairs, ~4x the GPU time for
zero new information) AND chain each curve point onto the previous one's Stage-B-tuned
weights instead of a clean Stage-A start, which would confound "did 122 help over 105" with
"the model already saw 105 pages of drift before 122 started". Instead:

  1. Stage A trains once. Its LoRA weights are saved with `peft.get_peft_model_state_dict`
     (adapter weights only, ~10-50 MB, not the frozen 350M-param base).
  2. For each curve point, a FRESH `LitComponent` is built (fresh base weights, freshly
     LoRA-wrapped) and Stage A's saved adapter state is loaded into it via
     `peft.set_peft_model_state_dict` before Stage B trains on that curve point's N pages.
     Four independent Stage-B fine-tunes of the same Stage-A start, which is what makes the
     105-vs-122 comparison (plan.md Step 28 point 2) mean what it's supposed to mean.

Resumable across Kaggle's ~9h interactive / ~12h commit ceiling (plan.md Step 28's own
timing note, and plan.md Step 11 point 8's "make the loop resumable" discipline applied
here to training instead of inference): every stage/curve-point boundary is written to
`data/models/ocr_lora/run_state.json` IMMEDIATELY, one point at a time, not batched at the
end --- so a second `kaggle kernels push` after a timeout reads that file first and skips
whatever it already marks done, rather than re-training from zero.

Usage (Kaggle, GPU, from the repo root, `configs/train_ocr.yaml` UNMODIFIED on disk):
    python scripts/run_finetune.py
Measure real per-step GPU time before committing the full curve (plan.md Step 28's own
"measure, don't assume" instruction, same discipline Step 18b should have applied first):
    python scripts/run_finetune.py --measure
Local dry run (CPU, tiny, proves the control flow only --- NOT a real fine-tune):
    python scripts/run_finetune.py --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doc_agent.data.validate import VAL_CHAPTERS  # noqa: E402
from doc_agent.eval.metrics import exact_formula_match, extract_formulas, ocr_f1  # noqa: E402
from doc_agent.logging_conf import get_logger  # noqa: E402
from doc_agent.training.datamodule import (  # noqa: E402
    DocDataModule,
    SeedByEpochCallback,
    make_collate_fn,
)
from doc_agent.training.lit_modules import LitComponent  # noqa: E402
from doc_agent.training.train import _build_trainer  # noqa: E402
from doc_agent.vision.ocr import (  # noqa: E402
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    _failure_reason,
)

logger = get_logger(__name__)

CURVE_POINTS: tuple[int, ...] = (25, 50, 105, 122)
MODELS_DIR = Path("data/models/ocr_lora")
STATE_PATH = MODELS_DIR / "run_state.json"
STAGE_A_CKPT = MODELS_DIR / "stage_a_adapter.pt"
CURVE_FIG_PATH = Path("reports/figures/step28_learning_curve.png")

# Step 18b's own found glyph-confusion class on this typeface, spot-checked here per
# plan.md Step 28's explicit instruction ("grep a handful of validation pages for these
# specific substitutions... before and after each curve point"). A rough presence-count
# proxy, not a token-aligned diff --- consistent with this project's other documented
# heuristics (see vision/ocr.py's _split_markdown_to_regions docstring on why an
# approximation is written up as one, not disguised as an exact measurement).
_NU_RE = "\\nu"
_VEC_RE = "\\vec{"
_ADVANCED_CONSTRUCTS = ("\\sqrt", "\\sum", "\\int", "\\prod")


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    """Write-then-rename so a mid-write interruption (Kaggle session death) can never
    leave `run_state.json` half-written and unreadable by the next resumed push."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def _load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"stage_a_done": False, "curve_points": {}}


def _load_val_records(val_dir: str) -> list[dict[str, Any]]:
    """Gold `{page_id, image_path, text}` rows for the 20 A&S validation pages, read
    directly rather than through `_ASStageBDataset` --- eval needs the raw gold text and
    the un-collated image, neither of which that dataset's `__getitem__` exposes."""
    from doc_agent.ingest.loader import _chapter_of

    records = []
    for jp in sorted(Path(val_dir).glob("*.json")):
        row = json.loads(jp.read_text(encoding="utf-8"))
        png_path = jp.with_suffix(".png")
        if not png_path.exists():
            raise FileNotFoundError(
                f"{jp} has no sibling image {png_path} -- run "
                "`ANNOT=1 bash scripts/get_data.sh` first"
            )
        actual = _chapter_of(row["printed_page"])
        if actual not in VAL_CHAPTERS:
            raise ValueError(f"LEAK — {row['page_id']} (chapter {actual}) is not a VAL chapter")
        records.append({"page_id": row["page_id"], "image_path": str(png_path), "text": row["text"]})
    return records


def _generate_page(lit: LitComponent, image: Any, device: str, max_new_tokens: int) -> str:
    """One Nougat forward pass through the currently-loaded (possibly LoRA-tuned) model.

    Mirrors `vision.ocr.Reader._generate()`'s exact decoding parameters (same
    max_new_tokens/repetition_penalty/bad_words_ids) so a curve point's validation score is
    comparable to the baseline numbers Step 16/18b/21 already measured with that reader ---
    a different decoding config would make "did fine-tuning help" partly a decoding-config
    question instead of a model-quality one.
    """
    import torch

    pixel_values = lit.processor(image, return_tensors="pt").pixel_values.to(device)
    lit.model.eval()
    with torch.no_grad():
        # Generic `PeftModel` (adapt.py's get_peft_model call has no task_type, so this is
        # not a PeftModelForSeq2SeqLM with its own .generate) forwards unknown attributes to
        # the wrapped base model via __getattr__ delegation -- standard, widely-relied-on
        # peft behavior, but not exercised anywhere in Step 27 (training only ever calls
        # forward(), never generate()). Fall back to the explicit path if delegation ever
        # doesn't resolve, rather than crashing the whole curve point on an AttributeError.
        try:
            generate_fn = lit.model.generate
        except AttributeError:
            generate_fn = lit.model.base_model.model.generate
        outputs = generate_fn(
            pixel_values,
            min_length=1,
            max_new_tokens=max_new_tokens,
            bad_words_ids=[[lit.processor.tokenizer.unk_token_id]],
            repetition_penalty=REPETITION_PENALTY,
        )
    text = lit.processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return lit.processor.post_process_generation(text, fix_markdown=False)


def _evaluate_on_val(
    lit: LitComponent, val_records: list[dict[str, Any]], device: str, max_new_tokens: int
) -> dict[str, Any]:
    """Run the just-trained model over all 20 val pages and score it the same way Step 29
    will score the 39 test pages (plan.md's own weighting note): failure rate as its own
    headline number, char-F1 + formula-weighted exact-match among the pages that produced
    output, plus the two Step 18b/28-flagged spot-checks (glyph confusion, advanced-
    construct lag)."""
    from PIL import Image as PILImage

    n = len(val_records)
    n_failed = 0
    f1s: list[float] = []
    exact_num = exact_den = 0.0
    adv_f1s: list[float] = []
    plain_f1s: list[float] = []
    nu_gold_total = nu_pred_v_total = 0
    hallucinated_vec_pages = 0
    per_page: list[dict[str, Any]] = []

    for rec in val_records:
        image = PILImage.open(rec["image_path"]).convert("RGB")
        pred = _generate_page(lit, image, device, max_new_tokens)
        gold = rec["text"]
        reason = _failure_reason(pred)
        # Save the raw prediction, not just its score -- found missing the hard way at
        # Step 28: without it, re-scoring past runs against a corrected `_failure_reason`
        # (as happened here, widened after `as_p0334`'s real spiral slipped through the
        # old 20-char unit cap) requires re-running generate() on every page instead of
        # just re-applying the fixed detector to text already on disk.
        row: dict[str, Any] = {
            "page_id": rec["page_id"], "failed": reason is not None, "pred_text": pred,
        }
        if reason is not None:
            n_failed += 1
            row["failure_reason"] = reason
        else:
            f1 = ocr_f1(pred, gold)
            f1s.append(f1)
            row["char_f1"] = f1
            gold_formulas = extract_formulas(gold)
            weight = len(gold_formulas)
            if weight:
                exact_num += exact_formula_match(pred, gold) * weight
                exact_den += weight
            has_adv = any(c in gold for c in _ADVANCED_CONSTRUCTS)
            (adv_f1s if has_adv else plain_f1s).append(f1)

            nu_gold_total += gold.count(_NU_RE)
            nu_pred_v_total += pred.count(" v ") + pred.count("v)") + pred.count("v(")
            if _VEC_RE in pred and _VEC_RE not in gold:
                hallucinated_vec_pages += 1

        per_page.append(row)
        status = f"FAILED:{reason}" if reason else f"f1={row.get('char_f1', 0):.3f}"
        logger.info(f"run_finetune: eval {rec['page_id']} -> {status}")

    return {
        "n_pages": n,
        "n_failed": n_failed,
        "failure_rate": n_failed / n,
        "mean_char_f1_among_successes": sum(f1s) / len(f1s) if f1s else 0.0,
        "formula_weighted_exact_match": (exact_num / exact_den) if exact_den else 0.0,
        "advanced_construct_mean_f1": sum(adv_f1s) / len(adv_f1s) if adv_f1s else None,
        "plain_mean_f1": sum(plain_f1s) / len(plain_f1s) if plain_f1s else None,
        "glyph_confusion_spotcheck": {
            "nu_occurrences_in_gold": nu_gold_total,
            "bare_v_occurrences_in_pred": nu_pred_v_total,
            "note": "rough presence-count proxy, not a token-aligned diff -- eyeball "
            "per_page below if this looks off",
        },
        "hallucinated_vec_pages": hallucinated_vec_pages,
        "per_page": per_page,
    }


def _save_adapter(lit: LitComponent, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lit.model.save_pretrained(str(out_dir))  # portable PEFT adapter dir: PeftModel.from_pretrained(base, out_dir)


def _no_ckpt_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """A cfg copy with `checkpoint.dir` stripped, so `_build_trainer`'s `ModelCheckpoint`
    callback is never added (its own guard is `if ckpt_dir:` -- see train.py).

    Real Kaggle run, 2026-08-11: `configs/train_ocr.yaml`'s `checkpoint.dir` is
    `data/models/ocr_lora` -- the SAME directory this script's own `_save_adapter` writes
    the small (~10-50 MB) LoRA-only adapter to. Left wired through to `_build_trainer`
    unchanged, Lightning's `ModelCheckpoint` saves a FULL checkpoint there too -- the
    entire 350M-param base model plus optimizer state, multiple GB, once per stage (Stage
    A + 4 curve points = 5x) -- which is what actually filled Kaggle's working disk and
    crashed the run (`OSError: No space left on device`) right after curve point n=25
    finished training, before its eval could even run. We never read that Lightning
    checkpoint (this script's own `run_state.json` + saved adapters ARE the resumable
    state), so it is pure waste here, not a safety net -- disable it entirely rather than
    trying to shrink or rotate it."""
    return {**cfg, "checkpoint": {}}


def _fresh_lit_from_stage_a(cfg: dict[str, Any], stage_a_ckpt: Path) -> LitComponent:
    """A new LitComponent (fresh base weights, freshly LoRA-wrapped) with Stage A's saved
    adapter weights loaded in -- the "clean restart per curve point" described in the
    module docstring."""
    import torch
    from peft import set_peft_model_state_dict

    lit = LitComponent(cfg, component="ocr")
    state_dict = torch.load(stage_a_ckpt, map_location="cpu")
    set_peft_model_state_dict(lit.model, state_dict)
    return lit


def _run_stage_a(cfg: dict[str, Any], collate_fn: Any, state: dict[str, Any]) -> None:
    import torch
    from peft import get_peft_model_state_dict

    if state["stage_a_done"] and STAGE_A_CKPT.exists():
        logger.info("run_finetune: Stage A already done (resumed) -- skipping")
        return

    logger.info("run_finetune: Stage A (NIST, shared pretrain) starting")
    lit = LitComponent(cfg, component="ocr")
    lit.set_stage(cfg["stage_a"])
    dm_a = DocDataModule(cfg, data_stage="nist", collate_fn=collate_fn)
    # Populate dm_a.train_dataset now (Trainer.fit() would call this itself, but
    # SeedByEpochCallback needs a direct reference to the dataset instance to mutate
    # before each epoch -- see that callback's docstring).
    dm_a.setup()
    seed_cb = SeedByEpochCallback(dm_a.train_dataset)
    trainer_a = _build_trainer(
        _no_ckpt_cfg(cfg), cfg["stage_a"], early_stopping=False, run_name="stage_a_nist",
        extra_callbacks=[seed_cb],
    )
    t0 = time.time()
    trainer_a.fit(lit, datamodule=dm_a)
    elapsed = time.time() - t0

    STAGE_A_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(lit.model), STAGE_A_CKPT)
    state["stage_a_done"] = True
    state["stage_a_elapsed_s"] = elapsed
    _atomic_write_json(STATE_PATH, state)
    logger.info(f"run_finetune: Stage A complete in {elapsed:.1f}s, adapter saved to {STAGE_A_CKPT}")


def _run_curve_point(
    cfg: dict[str, Any], collate_fn: Any, n_pages: int, val_records: list[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    key = str(n_pages)
    if state["curve_points"].get(key, {}).get("done"):
        logger.info(f"run_finetune: curve point n={n_pages} already done (resumed) -- skipping")
        return

    device = cfg.get("device", "cpu")
    logger.info(f"run_finetune: curve point n={n_pages} -- Stage B starting from Stage A weights")
    lit = _fresh_lit_from_stage_a(cfg, STAGE_A_CKPT)
    if device.startswith("cuda"):
        lit = lit.to(device)
    lit.set_stage(cfg["stage_b"])

    run_cfg = dict(cfg)
    run_cfg["data"] = {**cfg["data"], "stage_b_max_train_pages": n_pages}
    dm_b = DocDataModule(run_cfg, data_stage="as", collate_fn=collate_fn)
    trainer_b = _build_trainer(
        _no_ckpt_cfg(cfg), cfg["stage_b"], early_stopping=True, run_name=f"stage_b_n{n_pages}"
    )
    t0 = time.time()
    trainer_b.fit(lit, datamodule=dm_b)
    train_elapsed = time.time() - t0

    # Real Kaggle run, 2026-08-11: Trainer.fit()'s own teardown moved the LightningModule
    # back to CPU after training completed (confirmed by the crash this caused --
    # "Input type torch.cuda.FloatTensor and weight type torch.FloatTensor should be the
    # same" on the very next generate() call, right after an otherwise-clean 8-epoch
    # training run). Lightning does this to free GPU memory once a fit/test/predict call
    # ends; harmless for chained Trainer calls (each re-places the module before running),
    # but this script's eval loop calls generate() directly, outside any Trainer call, so
    # it must re-place the model itself rather than assume fit() left it where training put it.
    if device.startswith("cuda"):
        lit = lit.to(device)

    t0 = time.time()
    metrics = _evaluate_on_val(lit, val_records, device, cfg["data"]["max_target_length"])
    eval_elapsed = time.time() - t0

    adapter_dir = MODELS_DIR / f"curve_n{n_pages}"
    _save_adapter(lit, adapter_dir)

    state["curve_points"][key] = {
        "done": True,
        "n_train_pages": n_pages,
        "train_elapsed_s": train_elapsed,
        "eval_elapsed_s": eval_elapsed,
        "adapter_dir": str(adapter_dir),
        "val_metrics": metrics,
    }
    _atomic_write_json(STATE_PATH, state)
    logger.info(
        f"run_finetune: curve point n={n_pages} done -- failure_rate="
        f"{metrics['failure_rate']:.2f} char_f1={metrics['mean_char_f1_among_successes']:.3f} "
        f"(train {train_elapsed:.0f}s, eval {eval_elapsed:.0f}s)"
    )


def _plot_curve(state: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted(
        (int(k), v["val_metrics"]) for k, v in state["curve_points"].items() if v.get("done")
    )
    if len(points) < len(CURVE_POINTS):
        logger.warning(
            f"run_finetune: only {len(points)}/{len(CURVE_POINTS)} curve points done -- "
            "skipping the plot until the rest finish (re-run this script to continue)"
        )
        return

    ns = [p[0] for p in points]
    f1s = [p[1]["mean_char_f1_among_successes"] for p in points]
    fail_rates = [p[1]["failure_rate"] for p in points]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ns, f1s, "o-", color="tab:blue", label="mean char-F1 (successes)")
    ax1.set_xlabel("Stage B train pages")
    ax1.set_ylabel("char-F1", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(ns, fail_rates, "s--", color="tab:red", label="failure rate")
    ax2.set_ylabel("failure rate (of 20 val pages)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(0, 1)

    fig.suptitle("Step 28 learning curve — validation char-F1 & failure rate vs train-set size")
    fig.tight_layout()
    CURVE_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CURVE_FIG_PATH, dpi=150)
    logger.info(f"run_finetune: learning curve saved to {CURVE_FIG_PATH}")


def _measure(cfg: dict[str, Any], collate_fn: Any) -> None:
    """Real per-step GPU time on a short run, per plan.md Step 28's own instruction not to
    assume a number the way Step 18b's first full-book estimate did. Projects Stage A's
    full-epoch cost + one curve point's worst-case Stage-B cost + eval cost, then the
    likely total for all 4 curve points, so the caller can decide whether to split pushes
    BEFORE committing GPU hours to a run that might not fit Kaggle's ceiling."""
    device = cfg.get("device", "cpu")
    logger.info(f"run_finetune: --measure on device={device}")

    lit = LitComponent(cfg, component="ocr")
    if device.startswith("cuda"):
        lit = lit.to(device)
    lit.set_stage(cfg["stage_a"])
    dm_a = DocDataModule(cfg, data_stage="nist", collate_fn=collate_fn)
    trainer_a = _build_trainer(
        _no_ckpt_cfg(cfg), {**cfg["stage_a"], "max_steps": 5}, early_stopping=False,
        run_name="measure_stage_a",
    )
    t0 = time.time()
    trainer_a.fit(lit, datamodule=dm_a)
    stage_a_step_s = (time.time() - t0) / 5

    run_cfg = dict(cfg)
    run_cfg["data"] = {**cfg["data"], "stage_b_max_train_pages": 25}
    dm_b = DocDataModule(run_cfg, data_stage="as", collate_fn=collate_fn)
    trainer_b = _build_trainer(
        _no_ckpt_cfg(cfg),
        {**cfg["stage_b"], "max_steps": 5, "limit_val_batches": 1},
        early_stopping=False,
        run_name="measure_stage_b",
    )
    t0 = time.time()
    trainer_b.fit(lit, datamodule=dm_b)
    stage_b_step_s = (time.time() - t0) / 5

    # Same Trainer.fit() teardown behavior _run_curve_point hit for real -- re-place
    # before the manual generate() call below, don't assume fit() left it on device.
    if device.startswith("cuda"):
        lit = lit.to(device)

    from PIL import Image as PILImage

    val_records = _load_val_records(cfg["data"]["val_annot_dir"])
    sample_image = PILImage.open(val_records[0]["image_path"]).convert("RGB")
    t0 = time.time()
    _generate_page(lit, sample_image, device, cfg["data"]["max_target_length"])
    generate_s = time.time() - t0

    stage_a_full_steps = math.ceil(695 / int(cfg["stage_a"]["batch_size"]))
    stage_a_total_s = stage_a_full_steps * stage_a_step_s
    per_curve_point_worst_s = (
        max(CURVE_POINTS) * int(cfg["stage_b"]["max_epochs"]) * stage_b_step_s
        + 20 * generate_s
    )
    total_worst_s = stage_a_total_s + len(CURVE_POINTS) * per_curve_point_worst_s

    print("\n=== run_finetune --measure ===")
    print(f"  Stage A: {stage_a_step_s:.2f}s/step measured, {stage_a_full_steps} steps for the "
          f"full 695 pairs -> ~{stage_a_total_s/60:.1f} min")
    print(f"  Stage B: {stage_b_step_s:.2f}s/step measured (batch_size=1)")
    print(f"  eval generate(): {generate_s:.1f}s/page measured (max_new_tokens="
          f"{cfg['data']['max_target_length']})")
    print(f"  worst-case per curve point (max_epochs, no early stop, +20-page eval): "
          f"~{per_curve_point_worst_s/60:.1f} min")
    print(f"  worst-case TOTAL for Stage A + all 4 curve points: ~{total_worst_s/3600:.2f} h")
    print(f"  Kaggle ceiling (plan.md §11.4): ~9h interactive / ~12h commit")
    if total_worst_s > 9 * 3600:
        print("  -> projected total exceeds the interactive ceiling. Plan to run this via "
              "'Save & Run All (Commit)' and/or split curve points across multiple pushes "
              "(this script resumes from data/models/ocr_lora/run_state.json automatically).")
    else:
        print("  -> projected total fits inside a single interactive session, with margin.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", default="configs/train_ocr.yaml")
    p.add_argument("--measure", action="store_true", help="measure real step time, don't train")
    p.add_argument("--smoke", action="store_true", help="tiny CPU dry run of the control flow")
    args = p.parse_args()

    cfg = _load_yaml(args.cfg)

    import torch

    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    # W&B: online only if a key is actually available (Kaggle Secrets or env) -- a run must
    # not fail for lack of one, same rule train.py's own offline default already follows.
    # "disabled", not "offline", when there's no key: this script never reads the local
    # wandb logs (run_state.json is the real record), and "offline" mode still writes them
    # to disk per stage -- on the real Kaggle run this was one of several contributors to
    # the disk exhaustion that crashed the first push (see _no_ckpt_cfg's docstring for the
    # dominant cause). No local artifact we don't use is worth writing.
    cfg["logging"] = {
        **cfg.get("logging", {}),
        "wandb_mode": "online" if os.environ.get("WANDB_API_KEY") else "disabled",
    }

    if args.smoke:
        cfg["stage_a"] = {**cfg["stage_a"], "batch_size": 1, "max_steps": 3, "max_epochs": 1}
        cfg["stage_b"] = {
            **cfg["stage_b"], "batch_size": 1, "max_steps": 3, "max_epochs": 1,
            "limit_val_batches": 1,
        }
        cfg["data"] = {**cfg["data"], "max_target_length": 128}
        # Real bug, found auditing v5's downloaded output before committing it (2026-08-12):
        # MODELS_DIR itself was never redirected here, only STATE_PATH/STAGE_A_CKPT/
        # CURVE_FIG_PATH -- but `_run_curve_point`'s adapter_dir is built from MODELS_DIR,
        # so every --smoke self-check was writing real (tiny, junk) curve_n5/curve_n8
        # adapter directories straight into the committed data/models/ocr_lora/ path.
        # Redirecting MODELS_DIR too is what actually fixes it; the other three were
        # already correct.
        global CURVE_POINTS, MODELS_DIR, STATE_PATH, STAGE_A_CKPT, CURVE_FIG_PATH
        CURVE_POINTS = (5, 8)
        MODELS_DIR = Path("data/interim/smoke_run_finetune")
        STATE_PATH = MODELS_DIR / "run_state.json"
        STAGE_A_CKPT = MODELS_DIR / "stage_a_adapter.pt"
        CURVE_FIG_PATH = Path("data/interim/smoke_run_finetune/curve.png")

    lit0 = LitComponent(cfg, component="ocr")  # only to build the shared processor for collate_fn
    collate_fn = make_collate_fn(lit0.processor, cfg["data"]["max_target_length"])
    del lit0

    if args.measure:
        _measure(cfg, collate_fn)
        return

    logger.info(f"run_finetune: starting on device={cfg['device']}, "
                f"wandb_mode={cfg['logging']['wandb_mode']}, curve={CURVE_POINTS}")
    state = _load_state()
    _run_stage_a(cfg, collate_fn, state)

    val_records = _load_val_records(cfg["data"]["val_annot_dir"])
    if args.smoke:
        val_records = val_records[:2]  # keep the smoke run fast; not a real eval sample
    for n in CURVE_POINTS:
        _run_curve_point(cfg, collate_fn, n, val_records, state)

    _plot_curve(state)
    logger.info("run_finetune: all curve points complete")


if __name__ == "__main__":
    main()
