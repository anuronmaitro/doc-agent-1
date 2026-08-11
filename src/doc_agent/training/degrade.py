"""Training — Stage A synthetic degradation ops (Step 26, S2).

Pure image-processing functions extracted from `scripts/degrade_nist_pairs.py` unchanged
(same behavior, same config schema) so both that script AND `datamodule.py` (Step 27) share
ONE implementation instead of two copies drifting apart. The script uses these to
pre-render `reports/figures/` examples and an optional full materialized copy; the
datamodule (below) uses them to degrade a Stage A crop on-the-fly per sample, so training
never depends on a pre-generated `data/interim/nist_degraded/` directory existing.

Every op is randomized per-crop from a seeded RNG (config seed + pair index), so any single
crop's degraded output is independently reproducible without re-running the whole set --
see `degrade_one`.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def _sample(rng: np.random.Generator, bounds: list[float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi))


def pad_crop(img: np.ndarray, frac_of_size: float, fill: float) -> np.ndarray:
    """Pad with a flat fill matching *local* paper-white, not the page-wide mean.

    Tried `cv2.BORDER_REFLECT_101` first (continue the crop's own edge pixels) on the
    assumption that a crop's edge is blank background -- wrong: these crops are tightly
    bound to the formula's own bounding box (`crop_padding_pt: 3.0` in
    configs/nist_extract.yaml is only a few px), so text often sits right at the crop
    edge, and reflecting it mirrors real glyphs into the margin as garbled duplicate
    text. Caught by rendering the first example, not assumed -- reverted.

    A flat fill is the right approach, but the config's `target.mean_brightness`
    (~198, the *page-wide* average used for final calibration) is measurably darker
    than local whitespace immediately around an isolated formula (measured p90-p99
    brightness on real data/pages/*.png: ~210-219, since the page-wide mean is pulled
    down by denser ink regions elsewhere on the page). `fill` is therefore its own,
    separate config value (see configs/degradation.yaml "padding.fill"), not reused
    from `target.mean_brightness`.
    """
    h, w = img.shape[:2]
    pad_h = max(1, int(round(h * frac_of_size)))
    pad_w = max(1, int(round(w * frac_of_size)))
    return cv2.copyMakeBorder(
        img, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(float(fill),)
    )


def rotate(img: np.ndarray, angle_deg: float, fill: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(float(fill),)
    )


def elastic_warp(
    img: np.ndarray, alpha: float, sigma: float, rng: np.random.Generator, fill: float
) -> np.ndarray:
    h, w = img.shape[:2]
    dx = gaussian_filter((rng.random((h, w)) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((rng.random((h, w)) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([grid_y + dy, grid_x + dx])
    warped = map_coordinates(img, coords, order=1, mode="constant", cval=fill)
    return warped.astype(np.float32)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    ksize = max(3, int(round(sigma * 4)) | 1)  # odd kernel, scales with sigma
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def sensor_noise(img: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, std, size=img.shape)
    return img + noise


def paper_tone(img: np.ndarray, shift: float) -> np.ndarray:
    return img + shift


def uneven_illumination(img: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    # A smooth linear gradient in a random direction, +/- strength around 1.0.
    angle = rng.uniform(0, 2 * np.pi)
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    gradient = np.cos(angle) * xx + np.sin(angle) * yy
    gradient = gradient / (np.abs(gradient).max() + 1e-6)
    factor = 1.0 + strength * gradient
    return img * factor


def ink_bleed_or_thin(img: np.ndarray, kernel_radius: int, rng: np.random.Generator) -> np.ndarray:
    if kernel_radius <= 0:
        return img
    ksize = 2 * kernel_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    # Ink is dark (low value), so "bleed" (more ink) = erode the bright background =
    # cv2.erode on the raw image; "thinning" (less ink) = dilate the bright background =
    # cv2.dilate on the raw image.
    if rng.random() < 0.5:
        return cv2.erode(img, kernel)  # bleed
    return cv2.dilate(img, kernel)  # thin


def downsample_upsample(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(
        img, (max(1, int(w / factor)), max(1, int(h / factor))), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def calibrate(img: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    cur_mean, cur_std = float(img.mean()), float(img.std())
    if cur_std < 1e-6:
        return np.full_like(img, target_mean)
    rescaled = (img - cur_mean) * (target_std / cur_std) + target_mean
    return np.clip(rescaled, 0, 255)


def degrade_one(img: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Pad, apply the randomized op set, then calibrate to the A&S target stats.

    `img` is a single-channel (grayscale) float/uint array; `cfg` is
    `configs/degradation.yaml`'s loaded dict (or an equivalent in-memory dict, e.g. when
    called from the datamodule with a cached config rather than re-reading the file per
    sample). Returns a uint8 grayscale array, same convention as
    `scripts/degrade_nist_pairs.py`'s own output.
    """
    work = img.astype(np.float32)
    fill = float(cfg["padding"]["fill"])

    work = pad_crop(work, cfg["padding"]["frac_of_size"], fill)
    work = rotate(work, _sample(rng, cfg["rotation_deg"]), fill)
    work = elastic_warp(
        work,
        _sample(rng, cfg["elastic_warp"]["alpha"]),
        _sample(rng, cfg["elastic_warp"]["sigma"]),
        rng,
        fill,
    )
    work = gaussian_blur(work, _sample(rng, cfg["gaussian_blur_sigma"]))
    work = sensor_noise(work, _sample(rng, cfg["sensor_noise_std"]), rng)
    work = paper_tone(work, _sample(rng, cfg["paper_tone_shift"]))
    work = uneven_illumination(work, _sample(rng, cfg["uneven_illumination_strength"]), rng)
    work = ink_bleed_or_thin(work, int(round(_sample(rng, cfg["ink_bleed_or_thin_kernel"]))), rng)
    work = downsample_upsample(work, _sample(rng, cfg["downsample_factor"]))
    work = calibrate(work, cfg["target"]["mean_brightness"], cfg["target"]["contrast_std"])
    return work.astype(np.uint8)
