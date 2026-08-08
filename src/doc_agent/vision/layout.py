"""Stage 2 — layout detection / segmentation"""

from __future__ import annotations

from collections import Counter
from typing import Any

import cv2
import numpy as np

from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------------------------
# Column split. This corpus is set in two columns with a real central gutter: measured
# on the three A1 gold pages (printed 243/255/360), the column-ink minimum inside a
# center search band sits far below that page's median column ink (17.7k vs 38.2k
# median on p.360; 2.5k vs 27.0k on p.255; 4.4k vs 27.6k on p.243) — a real gap, not an
# assumed one. A page with no such gap (front matter, single-column chapter openers)
# is left as one column. MARGIN_TRIM_FRAC removes a measured scanner edge vignette
# (~1/40 of page width on every sampled page) that would otherwise register as ink.
# ------------------------------------------------------------------------------------
MARGIN_TRIM_FRAC = 0.03
GUTTER_SEARCH_FRAC = (0.40, 0.60)
GUTTER_RATIO = 0.5  # gutter ink must sit below half the page's median column ink

# ------------------------------------------------------------------------------------
# Line -> block segmentation, driven by each column's own row-ink profile (same idea as
# preprocess.py's blank-page threshold: derived from the page, not a fixed row count).
# LINE_GAP_MERGE_PX folds the few-pixel gaps inside one printed line (ascender/descender
# noise) into a single line band. A paragraph/table/section break is then whatever gap
# is unusually large *relative to that column's own median line gap*, since font size
# and line spacing both vary by chapter. BLOCK_GAP_FACTOR=1.8 was tried first and
# rejected: on the p.360 gold page the biggest real gap (61px) never exceeds 1.8x that
# column's own median gap (42px -> 75.6px), so the whole column never splits at all.
# 1.3x still clears the noise floor (line-internal gaps run 19-28px) while actually
# firing on the larger paragraph/table gaps that are really there (measured 42-61px).
# ------------------------------------------------------------------------------------
LINE_GAP_MERGE_PX = 14
BLOCK_GAP_FACTOR = 1.3
BLOCK_GAP_FLOOR_PX = 45

# ------------------------------------------------------------------------------------
# Table heuristic. A prose line is one continuous ink run; a numeric-table line is
# several short tokens whose start position repeats down the page (columns of numbers
# line up). So within a block: split each line into ink "segments" by its own
# horizontal gaps, then call the block a table when most lines carry several short
# segments AND a second segment start-x recurs across many lines — the signature of an
# aligned numeric grid, not flowing text.
# ------------------------------------------------------------------------------------
SEGMENT_GAP_PX = 18
TABLE_MIN_SEGMENTS_PER_LINE = 4
TABLE_MIN_LINE_FRACTION = 0.5
TABLE_COL_BIN_PX = 30

# Heading: a short block (<=2 lines) set in a visibly larger font than the rest of its
# column — section/chapter titles in this handbook are bigger, not just bold.
HEADING_MAX_LINES = 2
HEADING_HEIGHT_RATIO = 1.3

# Figure: a block that reads as almost no discrete text tokens but is mostly one large
# connected blob — a plot/diagram, not a paragraph or grid. Rare in this handbook (a
# few Bessel-function plots), so this only needs to catch the obvious cases.
FIGURE_MAX_MEDIAN_SEGMENTS = 1
FIGURE_MIN_BLOB_AREA_FRAC = 0.35


def _ink(gray: np.ndarray) -> np.ndarray:
    """Darkness per pixel, relative to this page's own ink/paper split (Otsu on the
    page's own histogram). A fixed or percentile baseline was tried first and rejected:
    on a page-height sum, even a few gray levels of baseline error leaks a small ink
    floor into *every* column, and a genuinely blank gutter — the one column that should
    read as ~0 — is exactly what that floor drowns out first. Otsu finds the page's own
    ink/paper split point, which on all three gold pages puts the true gutter at
    ratio < 0.03 against the page median instead of > 0.5 with a percentile baseline."""
    otsu_thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return np.clip(float(otsu_thr) - gray.astype(np.float32), 0, None)


def _bands_from_active(active: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(active)))
    return bands


def _merge_bands(bands: list[tuple[int, int]], gap_tol: float) -> list[tuple[int, int]]:
    if not bands:
        return []
    merged = [list(bands[0])]
    for s, e in bands[1:]:
        if s - merged[-1][1] <= gap_tol:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _group_lines_into_blocks(
    lines: list[tuple[int, int]], gap_tol: float
) -> list[list[tuple[int, int]]]:
    if not lines:
        return []
    groups = [[lines[0]]]
    for ln in lines[1:]:
        if ln[0] - groups[-1][-1][1] <= gap_tol:
            groups[-1].append(ln)
        else:
            groups.append([ln])
    return groups


def _find_column_split(ink: np.ndarray, x_lo: int, x_hi: int) -> int | None:
    """Return the gutter x, or None if this page reads as a single column."""
    w = x_hi - x_lo
    lo = x_lo + int(w * GUTTER_SEARCH_FRAC[0])
    hi = x_lo + int(w * GUTTER_SEARCH_FRAC[1])
    if hi <= lo:
        return None
    col_ink = ink[:, x_lo:x_hi].sum(axis=0)
    search = ink[:, lo:hi].sum(axis=0)
    gutter_rel = int(np.argmin(search))
    gutter_ink = float(search[gutter_rel])
    nonzero = col_ink[col_ink > 0]
    median_ink = float(np.median(nonzero)) if nonzero.size else 0.0
    if median_ink <= 0 or gutter_ink >= GUTTER_RATIO * median_ink:
        return None
    return lo + gutter_rel


def _line_segments(ink_row_slice: np.ndarray, x0: int) -> list[tuple[int, int]]:
    """Ink "words/numbers" on one line: horizontal ink runs split by SEGMENT_GAP_PX."""
    if ink_row_slice.size == 0:
        return []
    col = ink_row_slice.sum(axis=0)
    if col.max() <= 0:
        return []
    active = col > max(float(col.max()) * 0.02, 1.0)
    merged = _merge_bands(_bands_from_active(active), SEGMENT_GAP_PX)
    return [(x0 + s, x0 + e) for s, e in merged]


def _looks_like_table(
    lines: list[tuple[int, int]],
    per_line_segments: list[list[tuple[int, int]]],
    seg_counts: list[int],
) -> bool:
    if len(lines) < 3 or not seg_counts:
        return False
    if float(np.median(seg_counts)) < TABLE_MIN_SEGMENTS_PER_LINE:
        return False
    # A second (or later) token position that recurs at roughly the same x across many
    # lines is column alignment — the signature of a numeric grid, not flowing prose.
    bins: Counter[int] = Counter()
    for segs in per_line_segments:
        for s, _e in segs[1:]:
            bins[s // TABLE_COL_BIN_PX] += 1
    if not bins:
        return False
    return max(bins.values()) >= TABLE_MIN_LINE_FRACTION * len(lines)


def _looks_like_figure(
    col_slice: np.ndarray, lines: list[tuple[int, int]], median_segs: float
) -> bool:
    if median_segs > FIGURE_MAX_MEDIAN_SEGMENTS:
        return False
    y0, y1 = lines[0][0], lines[-1][1]
    block = col_slice[y0:y1, :]
    if block.size == 0 or block.max() <= 0:
        return False
    mask = (block > (float(block.max()) * 0.15)).astype(np.uint8)
    if not mask.any():
        return False
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return False
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return (largest / block.size) >= FIGURE_MIN_BLOB_AREA_FRAC


def _classify_block(
    col_slice: np.ndarray,
    lines: list[tuple[int, int]],
    x0: int,
    col_median_height: float,
) -> str:
    heights = [e - s for s, e in lines]
    per_line_segments = [_line_segments(col_slice[s:e, :], x0) for s, e in lines]
    seg_counts = [len(segs) for segs in per_line_segments]
    median_segs = float(np.median(seg_counts)) if seg_counts else 0.0

    if _looks_like_table(lines, per_line_segments, seg_counts):
        return "table"
    if _looks_like_figure(col_slice, lines, median_segs):
        return "figure"
    if len(lines) <= HEADING_MAX_LINES and float(np.median(heights)) >= (
        HEADING_HEIGHT_RATIO * col_median_height
    ):
        return "heading"
    return "text"


def _segment_column(page_id: str, ink: np.ndarray, x0: int, x1: int) -> list[Region]:
    """Segment one column into reading-order blocks, classified by kind."""
    if x1 <= x0:
        return []
    col_slice = ink[:, x0:x1]
    row_ink = col_slice.sum(axis=1)
    if row_ink.max() <= 0:
        return []
    thr = max(float(np.percentile(row_ink, 5)) * 3.0, float(row_ink.max()) * 0.003)
    lines = _merge_bands(_bands_from_active(row_ink > thr), LINE_GAP_MERGE_PX)
    if not lines:
        return []

    col_median_height = float(np.median([e - s for s, e in lines]))
    gaps = [lines[i + 1][0] - lines[i][1] for i in range(len(lines) - 1)]
    block_gap = (
        max(BLOCK_GAP_FACTOR * float(np.median(gaps)), BLOCK_GAP_FLOOR_PX)
        if gaps
        else (BLOCK_GAP_FLOOR_PX)
    )

    regions = []
    for block_lines in _group_lines_into_blocks(lines, block_gap):
        kind = _classify_block(col_slice, block_lines, x0, col_median_height)
        bbox = (x0, block_lines[0][0], x1, block_lines[-1][1])
        regions.append(Region(page_id=page_id, bbox=bbox, kind=kind))
    return regions


def _iou(a: tuple[int, int, int, int], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    return inter / (area_a + area_b - inter)


class _TatrRefiner:
    """Best-effort table confirmation via cfg["layout"]["model"] (microsoft/table-
    transformer-detection). The classical projection-profile method above is the
    primary, deterministic signal and needs no model or network; this only upgrades a
    block to "table" when TATR is independently confident a table sits there, and it
    never raises — if the model can't be loaded (no network, no cache, a CI sandbox),
    layout detection still works from the classical method alone."""

    def __init__(self, cfg: dict) -> None:
        layout_cfg = cfg.get("layout") or {}
        self.model_name: str = layout_cfg.get("model", "microsoft/table-transformer-detection")
        self.score_thr: float = float(layout_cfg.get("score_thr", 0.5))
        self.requested_device: str = str(cfg.get("device", "cpu"))
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"
        self._load_failed = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            import torch
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection

            self._device = self.requested_device
            if self._device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning(
                    "layout: cfg requests cuda but no GPU is visible; running TATR on CPU"
                )
                self._device = "cpu"
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
            model.eval()
            model.to(self._device)
            self._model = model
        except Exception:
            logger.warning(
                f"layout: could not load {self.model_name}; classical table detection only"
            )
            self._load_failed = True
            return False
        return True

    def refine(self, gray: np.ndarray, regions: list[Region]) -> list[Region]:
        if not regions or not self._ensure_loaded():
            return regions
        try:
            import torch
            from PIL import Image as PILImage

            rgb = PILImage.fromarray(gray).convert("RGB")
            inputs = self._processor(images=rgb, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            target_sizes = torch.tensor([rgb.size[::-1]])
            result = self._processor.post_process_object_detection(
                outputs, threshold=self.score_thr, target_sizes=target_sizes
            )[0]
            id2label = self._model.config.id2label
            table_boxes = [
                tuple(box.tolist())
                for box, label in zip(result["boxes"], result["labels"], strict=False)
                if id2label[int(label)] == "table"
            ]
        except Exception:
            logger.warning("layout: TATR inference failed on a page; keeping classical regions")
            return regions

        if not table_boxes:
            return regions
        out = []
        for r in regions:
            if r.kind != "table" and any(_iou(r.bbox, tb) >= 0.3 for tb in table_boxes):
                out.append(Region(page_id=r.page_id, bbox=r.bbox, kind="table"))
            else:
                out.append(r)
        return out


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions, in reading order (left column top-to-
    bottom, then right column top-to-bottom). Classical projection-profile segmentation
    does the real work; cfg["layout"]["model"] (TATR) optionally confirms extra tables
    when it's available, per cfg["layout"]["method"] = "projection-profile+tatr"."""
    layout_cfg = cfg.get("layout") or {}
    refiner = _TatrRefiner(cfg) if layout_cfg.get("model") else None

    regions: list[Region] = []
    for page in pages:
        gray = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(
                f"layout: could not read {page.image_path!r} for page {page.id!r}"
            )
        h, w = gray.shape
        margin = int(w * MARGIN_TRIM_FRAC)
        x_lo, x_hi = margin, w - margin
        ink = _ink(gray)
        gutter = _find_column_split(ink, x_lo, x_hi)

        if gutter is None:
            page_regions = _segment_column(page.id, ink, x_lo, x_hi)
        else:
            page_regions = _segment_column(page.id, ink, x_lo, gutter) + _segment_column(
                page.id, ink, gutter, x_hi
            )

        if refiner is not None:
            page_regions = refiner.refine(gray, page_regions)
        regions.extend(page_regions)

    return regions
