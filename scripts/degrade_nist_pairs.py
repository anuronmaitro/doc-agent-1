"""Step 26 — synthetic degradation pipeline.

Takes Step 25's clean NIST Stage A crops (data/annot/nist/, FINAL -- read-only input,
see plan.md Step 26's handoff-risk warning) and makes them look like the 1964 A&S scan,
so Step 27/28 can fine-tune the OCR reader on formula-level (image, latex) pairs that
actually resemble the real corpus, not clean pdfTeX renders.

Two permanent, on-purpose gaps inherited from Step 25 (not this step's job to fix, see
plan.md Step 26 and data/README.md "Step 25"): no stacked sums/integrals/products or
radicals anywhere in the 695 pairs, and imperfect overline accents on the few pairs that
have one. This script only changes how the crops LOOK, never their `text` label.

Ops, each randomized per-crop from a seeded per-sample RNG (config seed + pair index, so
any single crop's output is independently reproducible without re-running the whole set):

1. Pad with a plain white margin (fill = calibration target mean brightness) sized as a
   fraction of the crop's own dimensions -- crops are tightly bound to one formula's
   bounding box (measured: a few hundred px wide, commonly under 100px tall), so rotating
   or warping an unpadded crop clips glyphs right at the edge.
2. Rotate by a small random angle.
3. Elastic warp (smoothed random displacement field) -- mild handset/press irregularity.
4. Gaussian blur -- imperfect scan focus.
5. Sensor noise -- additive Gaussian noise.
6. Paper tone shift -- additive brightness shift before final calibration.
7. Uneven illumination -- a smooth multiplicative gradient (corner-to-corner falloff).
8. Ink bleed or thinning -- a random small morphological dilate (bleed) or erode (thin).
9. Downsample -> upsample -- simulates the resolution loss of print + scan at 300dpi.
10. Final linear calibration to the measured A&S target mean/contrast (configs/degradation.yaml
    "target", verified 2026-08-11 over data/pages/*.png -- see that file's comment).

The op implementations themselves (pad_crop ... degrade_one) moved to
`doc_agent.training.degrade` at Step 27, so `training/datamodule.py` can degrade a Stage A
crop on-the-fly per training sample using the EXACT same code this script uses to render
`reports/figures/`'s examples, instead of a second copy drifting out of sync. This script's
own behavior/CLI/config schema is unchanged by that move.

Usage:
    python scripts/degrade_nist_pairs.py --cfg configs/degradation.yaml
    python scripts/degrade_nist_pairs.py --cfg configs/degradation.yaml --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doc_agent.training.degrade import degrade_one, pad_crop  # noqa: E402


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_pairs(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _side_by_side(original: np.ndarray, degraded: np.ndarray, caption: str) -> Image.Image:
    h = max(original.shape[0], degraded.shape[0])
    gap = 12
    band = 22
    w = original.shape[1] + degraded.shape[1] + gap
    canvas = Image.new("L", (w, h + band), color=255)
    canvas.paste(Image.fromarray(original), (0, band))
    canvas.paste(Image.fromarray(degraded), (original.shape[1] + gap, band))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover - font loading never fails in practice
        font = None
    draw.text((2, 2), caption, fill=0, font=font)
    return canvas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", default="configs/degradation.yaml")
    p.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    args = p.parse_args()

    cfg = load_config(args.cfg)
    pairs = load_pairs(cfg["pairs_path"])
    if args.limit:
        pairs = pairs[: args.limit]

    out_dir = Path(cfg["out_dir"])
    images_out = out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    examples_dir = Path(cfg["examples_dir"])
    examples_dir.mkdir(parents=True, exist_ok=True)

    n_examples = cfg["n_examples"]
    example_indices = (
        set(np.linspace(0, len(pairs) - 1, n_examples).astype(int)) if pairs else set()
    )

    manifest_path = out_dir / "degraded_pairs.jsonl"
    means, stds = [], []
    example_count = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for i, rec in enumerate(pairs):
            src = Image.open(rec["image"]).convert("L")
            arr = np.asarray(src, dtype=np.float32)
            rng = np.random.default_rng(cfg["seed"] + i)
            degraded = degrade_one(arr, cfg, rng)
            means.append(float(degraded.mean()))
            stds.append(float(degraded.std()))

            out_path = images_out / f"{rec['pair_id']}.png"
            Image.fromarray(degraded).save(out_path)
            manifest.write(
                json.dumps(
                    {
                        "pair_id": rec["pair_id"],
                        "eqn_id": rec["eqn_id"],
                        "text": rec["text"],
                        "image": str(out_path).replace("\\", "/"),
                        "source_image": rec["image"],
                    }
                )
                + "\n"
            )

            if i in example_indices:
                example_count += 1
                padded_original = pad_crop(
                    arr, cfg["padding"]["frac_of_size"], cfg["padding"]["fill"]
                ).astype(np.uint8)
                caption = f"{rec['pair_id']}  (left: padded original, right: degraded)"
                comparison = _side_by_side(padded_original, degraded, caption)
                comparison.save(examples_dir / f"step26_degradation_example_{example_count}.png")

    print(f"pairs processed: {len(pairs)}")
    print(f"degraded images: {out_dir}/images/ (+ {manifest_path.name})")
    print(f"examples written: {example_count} -> {examples_dir}/")
    if means:
        print(
            f"degraded set stats: mean={sum(means) / len(means):.2f} "
            f"(target {cfg['target']['mean_brightness']}), "
            f"std={sum(stds) / len(stds):.2f} (target {cfg['target']['contrast_std']})"
        )


if __name__ == "__main__":
    main()
