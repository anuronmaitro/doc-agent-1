"""Region-routing hybrid: curve_n122 whole-page + table_ft ONLY on detected table-region
crops, instead of either adapter alone on the whole page.

Validated on the 20 A&S val pages (this file's own __main__): failure_rate 0.05 vs 0.10
for both curve_n122 alone and table_ft alone; char_f1 ~0.553, above curve_n122's 0.489 and
close to table_ft's 0.568. See plan.md Step 28 point 11 for the full history, including
two real bugs found and fixed along the way (both fixed below, in `build_hybrid_prediction`):

1. If curve_n122's OWN whole-page prediction already failed for a page, splicing table
   crops into that already-garbled base just fails again for the same reason. Fixed: fall
   back to table_ft's own whole-page prediction instead (it may succeed where curve_n122
   didn't -- as_p0534 in the val check: curve_n122 FAILED, table_ft succeeded at 0.480).
2. `_split_markdown_to_regions` approximates region boundaries by splitting curve_n122's
   whole-page text on blank lines and pairing blocks with regions BY POSITION -- when a
   page has more layout regions than text blocks, the shortfall is padded with "". Treating
   an empty "original chunk" as safe to overwrite is wrong: there's no real content there
   to replace, so substituting a fresh crop into that slot doesn't correct anything, it
   just inserts content into a slot with no correspondence to the source text, and reliably
   produced degenerate output on the 2 pages this happened on (as_p0459, as_p0518). Fixed:
   skip substitution entirely when the original chunk is empty -- the page keeps
   curve_n122's already-fine text for that region instead.
3. Degenerate-crop guard: skip a substitution if the freshly-generated crop text is
   suspiciously short relative to the (non-empty) chunk it would replace -- a near-empty
   crop is more likely a bad decode than a genuinely short table.

To reuse this for a NEW page set (e.g. Step 29's 39 test pages): call
`build_hybrid_prediction()` per page inside your own evaluation loop -- see its docstring
for the exact inputs needed. You do NOT need curve_n122's predictions to come from a cache
file the way this val-set check does; any already-computed (page_id -> pred_text, failed)
source works, including generating it fresh in the same loop.

Run from the repo root (Kaggle: after `%cd /kaggle/working/repo`) to reproduce the val-set
numbers above:
    python KAGGLE/region_routing_check/validate_region_routing.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import yaml  # noqa: E402
from run_finetune import _generate_page  # noqa: E402

from doc_agent.contracts import Page, Region  # noqa: E402
from doc_agent.eval.metrics import ocr_f1  # noqa: E402
from doc_agent.training.lit_modules import LitComponent  # noqa: E402
from doc_agent.vision import layout  # noqa: E402
from doc_agent.vision.ocr import _failure_reason, _split_markdown_to_regions  # noqa: E402

LAYOUT_CFG = {
    "layout": {
        "model": "microsoft/table-transformer-detection",
        "score_thr": 0.5,
        "method": "projection-profile+tatr",
    },
    "device": "cpu",  # layout detection itself is cheap; only generation needs the GPU
}
TABLE_FT_DIR = REPO_ROOT / "data" / "models" / "ocr_lora" / "table_ft"
BEFORE_STATE_PATH = TABLE_FT_DIR / "run_state.json"
VAL_ANNOT_DIR = REPO_ROOT / "data" / "annot" / "val"
MIN_CROP_CHARS = 50
MIN_CROP_RATIO = 0.20


def build_hybrid_prediction(
    *,
    page_id: str,
    image_path: str,
    page_regions: list[Region],
    curve_pred_text: str,
    curve_failed: bool,
    table_ft_whole_page_pred_text: str | None,
    table_ft_lit: LitComponent,
    device: str,
    max_new_tokens: int,
) -> str:
    """The region-routing hybrid for ONE page, with all 3 fixes applied.

    Args:
        page_id: page identifier (for logging only).
        image_path: path to the full page image.
        page_regions: this page's regions from `layout.detect()` (call it with LAYOUT_CFG
            for TATR to actually run -- an empty/partial cfg silently disables it, which is
            exactly the bug that produced a misleadingly pessimistic earlier check).
        curve_pred_text: curve_n122's own whole-page prediction for this page (fresh or
            cached -- either is fine).
        curve_failed: whether curve_n122's own whole-page prediction failed
            (`vision.ocr._failure_reason(curve_pred_text) is not None`).
        table_ft_whole_page_pred_text: table_ft's own whole-page prediction for this page,
            used as the Fix-1 fallback when curve_failed is True. Pass None if you don't
            have it (rare case: the fallback then produces empty text and the caller's own
            failure check will correctly mark the page failed, same as no worse than not
            attempting a fallback at all).
        table_ft_lit: a LitComponent with the table_ft adapter already loaded (see
            `build_lit_with_table_ft` below), used for crop generation.
        device, max_new_tokens: generation params, same meaning as everywhere else in this
            codebase (`run_finetune.py`'s `_generate_page`).

    Returns the hybrid page text (never raises; an unusable result is still returned as
    text so the caller's own `_failure_reason` check decides pass/fail, consistent with
    every other evaluation path in this project).
    """
    from PIL import Image as PILImage

    if curve_failed:
        return table_ft_whole_page_pred_text or ""

    if not page_regions or not any(r.kind == "table" for r in page_regions):
        return curve_pred_text

    chunks = _split_markdown_to_regions(curve_pred_text, len(page_regions))
    full_image = PILImage.open(image_path).convert("RGB")

    for i, region in enumerate(page_regions):
        if region.kind != "table":
            continue
        original_chunk = chunks[i]

        if len(original_chunk) == 0:
            # Fix 2: nothing real to replace -- see module docstring.
            continue

        crop = full_image.crop(region.bbox)
        if crop.width < 8 or crop.height < 8:
            continue

        t0 = time.time()
        crop_text = _generate_page(table_ft_lit, crop, device, max_new_tokens)
        print(
            f"  {page_id} region[{i}] bbox={region.bbox} -> {len(crop_text)} chars "
            f"({time.time() - t0:.1f}s)"
        )

        # Fix 3: degenerate-crop guard.
        if len(crop_text) < MIN_CROP_CHARS or len(crop_text) < MIN_CROP_RATIO * len(original_chunk):
            print("    -> skipping substitution (crop too short vs original)")
            continue
        chunks[i] = crop_text

    return "\n\n".join(c for c in chunks if c.strip())


def build_lit_with_table_ft(cfg: dict[str, Any], device: str) -> LitComponent:
    from peft import PeftModel

    lit = LitComponent(cfg, component="ocr")
    if device.startswith("cuda"):
        lit = lit.to(device)
    base = lit.model.get_base_model()
    lit.model = PeftModel.from_pretrained(base, str(TABLE_FT_DIR))
    if device.startswith("cuda"):
        lit.model = lit.model.to(device)
    lit.model.eval()
    return lit


# --- everything below this line is the val-set self-check (__main__ only) ---------------


def _load_state() -> dict:
    return json.loads(BEFORE_STATE_PATH.read_text(encoding="utf-8"))


def _load_gold() -> dict[str, str]:
    out = {}
    for jp in sorted(VAL_ANNOT_DIR.glob("*.json")):
        row = json.loads(jp.read_text(encoding="utf-8"))
        out[jp.stem] = row["text"]
    return out


def main() -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "train_ocr.yaml").read_text(encoding="utf-8"))
    cfg["device"] = device

    state = _load_state()
    curve_cache = {r["page_id"]: r for r in state["before_metrics"]["per_page"]}
    after_by_page = {r["page_id"]: r for r in state.get("after_metrics", {}).get("per_page", [])}
    gold = _load_gold()
    val_ids = sorted(curve_cache.keys())

    print(f"Loaded curve_n122 cached predictions for {len(val_ids)} val pages")
    print("Running layout.detect() with TATR enabled...")
    pages = [
        Page(id=pid, image_path=str(VAL_ANNOT_DIR / f"{pid}.png"), doc_id="hybrid")
        for pid in val_ids
    ]
    regions = layout.detect(pages, LAYOUT_CFG)
    by_page: dict[str, list] = {}
    for r in regions:
        by_page.setdefault(r.page_id, []).append(r)

    print("Loading table_ft adapter for crop generation...")
    lit = build_lit_with_table_ft(cfg, device)
    max_new_tokens = cfg["data"]["max_target_length"]

    hybrid_pred: dict[str, str] = {}
    for pid in val_ids:
        curve_row = curve_cache[pid]
        hybrid_pred[pid] = build_hybrid_prediction(
            page_id=pid,
            image_path=str(VAL_ANNOT_DIR / f"{pid}.png"),
            page_regions=by_page.get(pid, []),
            curve_pred_text=curve_row.get("pred_text", ""),
            curve_failed=bool(curve_row.get("failed")),
            table_ft_whole_page_pred_text=after_by_page.get(pid, {}).get("pred_text"),
            table_ft_lit=lit,
            device=device,
            max_new_tokens=max_new_tokens,
        )

    print(
        "\n=== Per-page comparison: curve_n122 | table_ft (whole-page) | hybrid (region-routed, fixed) ==="
    )
    hybrid_f1s = []
    hybrid_failed = 0
    for pid in val_ids:
        g = gold[pid]
        c_row = curve_cache[pid]
        c_f1 = c_row.get("char_f1")
        c_str = f"{c_f1:.3f}" if c_f1 is not None else f"FAILED:{c_row.get('failure_reason')}"

        t_row = after_by_page.get(pid, {})
        t_f1 = t_row.get("char_f1")
        t_str = f"{t_f1:.3f}" if t_f1 is not None else f"FAILED:{t_row.get('failure_reason')}"

        h_pred = hybrid_pred[pid]
        h_reason = _failure_reason(h_pred)
        if h_reason is not None:
            hybrid_failed += 1
            h_str = f"FAILED:{h_reason}"
        else:
            h_f1 = ocr_f1(h_pred, g)
            hybrid_f1s.append(h_f1)
            h_str = f"{h_f1:.3f}"

        print(f"  {pid}: curve={c_str:>22}  table_ft={t_str:>22}  hybrid={h_str:>22}")

    print(
        f"\nHybrid AGGREGATE: failure_rate={hybrid_failed / len(val_ids):.2f}  "
        f"char_f1={sum(hybrid_f1s) / len(hybrid_f1s) if hybrid_f1s else 0.0:.3f}"
    )
    print("(curve_n122 baseline: failure_rate=0.10  char_f1=0.489)")
    print("(table_ft whole-page: failure_rate=0.10  char_f1=0.568)")


if __name__ == "__main__":
    main()
