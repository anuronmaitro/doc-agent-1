"""Stage 1 — deskew / denoise / binarize / augment"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# Processed pages land here. Already gitignored (.gitignore line 2) — these are derived
# artifacts, recreated by re-running the pipeline, never committed.
INTERIM_DIR = Path("data/interim")

# Written alongside the images so a re-run can tell "already done with THESE settings"
# from "done with different settings" — see _is_current() / run().
MANIFEST_NAME = "_preprocess_manifest.json"

# --- defaults, every one chosen from a measurement on our own corpus ----------------------
# The numbers quoted below were measured on data/pages/ at 300 dpi; tests/test_ingest.py
# pins the behaviour they buy (dead zone, no binarisation, no saturation).

# Deskew search: our scan is near-perfectly aligned, so a +-3 deg window is already
# generous. Estimated on a 0.5-scale copy: accurate to <=0.05 deg for |angle| >= 0.5 deg,
# and it floors to 0 below ~0.25 deg, which is well inside the dead zone below.
DESKEW_LIMIT_DEG = 3.0
DESKEW_COARSE_STEP_DEG = 0.2
DESKEW_FINE_STEP_DEG = 0.05
DESKEW_SCALE = 0.5

# Rotation resamples every pixel, and interpolation blurs exactly the thin sub/superscript
# strokes this corpus is full of -- but a real minority of pages are genuinely crooked.
# Measured over all 1040 content pages: 57 (5.5%) sit at or above this threshold, the worst
# at 2.65 deg; the other 94.5% are already straight. So we rotate only those 57 and leave
# ~983 pages untouched, instead of paying interpolation cost on the whole book. Measured
# after the run, residual skew on the corrected pages is max 0.50 deg, mean 0.054 deg.
DESKEW_MIN_ANGLE_DEG = 0.10

# Light bilateral: edge-preserving, so it attenuates paper grain without rounding off
# stroke ends the way a Gaussian does. Measured on p.360 (dense two-column formula),
# mean gradient on stroke cores: original 339.9 -> bilateral(5,15,15) 336.4 (-1.0%),
# vs Gaussian 3x3 275.4 (-19%) and median 3x3 322.1 (-5%). Chosen as the gentlest
# option that still is a real filter.
DENOISE_DIAMETER = 5
DENOISE_SIGMA_COLOR = 15
DENOISE_SIGMA_SPACE = 15

# CLAHE normalises contrast across pages whose exposure varies (measured page means run
# 188-209, stds 7-47), so the reader sees consistent input. tile=16 beat tile=8 on stroke
# gradient at every clip level tested; clip>2 was rejected because it thickens strokes
# (ink coverage on p.255 went 4.65% -> 6.30% at clip=4, the same failure mode that makes
# hard binarisation unusable here). At clip=2/tile=16 stroke gradient is preserved or
# slightly improved (p.360 339.9->338.9, p.243 347.4->361.3, p.255 300.7->305.0).
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = 16

# NO binarisation anywhere in this module, by design. Output stays 8-bit grayscale:
# thresholding thickens thin sub/superscripts and destroys the antialiased stroke edges
# the OCR reader depends on (summary.md 7a: "classical deskew/denoise/CLAHE only").


def _params(cfg: dict) -> dict:
    """Resolve preprocessing parameters.

    Reads an optional `preprocess:` block from configs/config.yaml, falling back to the
    measured defaults above. The block is deliberately absent from config.yaml today:
    Step 02 fixed that file's contents and the form's Section 4 table must match it
    character-for-character (summary.md 7a), so this stage stays configurable without
    forcing an edit that would desync the two.
    """
    p = dict(cfg.get("preprocess") or {})
    return {
        "deskew": bool(p.get("deskew", True)),
        "deskew_min_angle_deg": float(p.get("deskew_min_angle_deg", DESKEW_MIN_ANGLE_DEG)),
        "denoise": bool(p.get("denoise", True)),
        "denoise_diameter": int(p.get("denoise_diameter", DENOISE_DIAMETER)),
        "denoise_sigma_color": int(p.get("denoise_sigma_color", DENOISE_SIGMA_COLOR)),
        "denoise_sigma_space": int(p.get("denoise_sigma_space", DENOISE_SIGMA_SPACE)),
        "clahe": bool(p.get("clahe", True)),
        "clahe_clip_limit": float(p.get("clahe_clip_limit", CLAHE_CLIP_LIMIT)),
        "clahe_tile_grid": int(p.get("clahe_tile_grid", CLAHE_TILE_GRID)),
        "seed": int(cfg.get("seed", 42)),
    }


def _estimate_skew_deg(gray: np.ndarray) -> float:
    """Angle (degrees) this page must be rotated by to sit straight.

    Projection-profile method: text lines produce a horizontal ink profile whose
    row-to-row differences are sharpest when the lines are level, so we search the
    rotation that maximises the variance of that difference. Chosen over Hough lines
    because this corpus is two-column mathematics — it has few long straight rules to
    detect, but very regular text baselines.

    Validated by injecting known rotations into a real page and recovering them:
    -1.00 -> +1.00, -0.50 -> +0.50, +0.50 -> -0.50, +1.00 -> -1.00 (error <= 0.05 deg).
    """
    small = cv2.resize(gray, None, fx=DESKEW_SCALE, fy=DESKEW_SCALE, interpolation=cv2.INTER_AREA)
    ink = 255.0 - small.astype(np.float32)
    # Keep only the darkest quarter: drops paper tone so the profile tracks glyphs, not shading.
    ink = np.clip(ink - float(np.percentile(ink, 75)), 0, None)
    h, w = ink.shape
    centre = (w / 2.0, h / 2.0)

    def sharpness(angle: float) -> float:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        # default borderValue is 0 == "no ink", which is what we want outside the page
        rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR)
        return float(np.var(np.diff(rotated.sum(axis=1))))

    limit = DESKEW_LIMIT_DEG
    coarse = float(max(np.arange(-limit, limit + 1e-9, DESKEW_COARSE_STEP_DEG), key=sharpness))
    fine_grid = np.arange(
        coarse - DESKEW_COARSE_STEP_DEG,
        coarse + DESKEW_COARSE_STEP_DEG + 1e-9,
        DESKEW_FINE_STEP_DEG,
    )
    return float(max(fine_grid, key=sharpness))


def _deskew(gray: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
    """Rotate the page level, but only when it is measurably crooked."""
    angle = _estimate_skew_deg(gray)
    if abs(angle) < params["deskew_min_angle_deg"]:
        return gray, angle  # inside the dead zone: leave the pixels alone
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    # BORDER_REPLICATE, not a black fill: a dark wedge in the margin would read as ink
    # to the layout stage and could be picked up as a spurious region.
    rotated = cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def _denoise(gray: np.ndarray, params: dict) -> np.ndarray:
    return cv2.bilateralFilter(
        gray,
        params["denoise_diameter"],
        params["denoise_sigma_color"],
        params["denoise_sigma_space"],
    )


def _clahe(gray: np.ndarray, params: dict) -> np.ndarray:
    tile = params["clahe_tile_grid"]
    clahe = cv2.createCLAHE(clipLimit=params["clahe_clip_limit"], tileGridSize=(tile, tile))
    return clahe.apply(gray)


def _process_one(gray: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
    """Deskew -> denoise -> CLAHE, in that order.

    Geometry first, so later filters act on final pixel positions. Denoise before CLAHE
    because CLAHE amplifies local contrast and would amplify grain along with it.
    """
    angle = 0.0
    out = gray
    if params["deskew"]:
        out, angle = _deskew(out, params)
    if params["denoise"]:
        out = _denoise(out, params)
    if params["clahe"]:
        out = _clahe(out, params)
    return out, angle


def _is_current(manifest_path: Path, params: dict) -> bool:
    """True if data/interim/ was already built with exactly these parameters."""
    if not manifest_path.exists():
        return False
    try:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(prior.get("params") == params)


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Classical preprocessing: deskew, light denoise, CLAHE contrast. Grayscale in, grayscale out.

    Writes one PNG per page to data/interim/ and returns new `Page` objects with
    `image_path` repointed there; `id` and `doc_id` are carried through unchanged so the
    chapter-level split (summary.md 3f) survives the stage.

    Idempotent: a page is re-processed only if its output is missing or the parameters
    changed since the last run (tracked in data/interim/_preprocess_manifest.json), so
    re-running is cheap and an interrupted run resumes instead of restarting.

    Deterministic: every operation here is a fixed convolution or affine transform, so
    there is no sampling to seed. `cfg["seed"]` is still recorded in the manifest, both
    to pin the run and so that adding any stochastic augmentation later cannot silently
    become unreproducible.
    """
    if not pages:
        return []

    params = _params(cfg)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = INTERIM_DIR / MANIFEST_NAME
    params_current = _is_current(manifest_path, params)

    out_pages: list[Page] = []
    processed = skipped = rotated = 0
    angles: list[float] = []
    t0 = time.time()

    for page in pages:
        dst = INTERIM_DIR / f"{page.id}.png"
        if params_current and dst.exists():
            skipped += 1
        else:
            src = Path(page.image_path)
            gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(
                    f"preprocess: could not read {src} for page {page.id!r}. "
                    "Run `bash scripts/get_data.sh` to rebuild data/pages/."
                )
            out, angle = _process_one(gray, params)
            if not cv2.imwrite(str(dst), out):
                raise OSError(f"preprocess: failed to write {dst}")
            processed += 1
            angles.append(angle)
            if abs(angle) >= params["deskew_min_angle_deg"]:
                rotated += 1
            if processed % 100 == 0:
                rate = processed / max(time.time() - t0, 1e-9)
                logger.info(f"preprocess: {processed} pages written ({rate:.1f}/s)")

        out_pages.append(
            Page(
                id=page.id,
                image_path=str(dst).replace("\\", "/"),
                doc_id=page.doc_id,
            )
        )

    manifest = {
        "params": params,
        "n_pages": len(out_pages),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    max_abs_angle = max((abs(a) for a in angles), default=0.0)
    logger.info(
        f"preprocess: {len(out_pages)} pages -> {INTERIM_DIR} "
        f"({processed} processed, {skipped} already current, "
        f"{rotated} deskewed, max |angle| {max_abs_angle:.2f} deg)"
    )
    return out_pages
