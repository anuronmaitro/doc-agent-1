"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from ..contracts import *  # noqa
from ..ingest.loader import _chapter_of
from ..logging_conf import get_logger

logger = get_logger(__name__)

# Where a page's own rendered image lives, by convention (not part of cfg): preprocess.py
# (Step 9) writes the deskewed/denoised/CLAHE'd version to data/interim/<page_id>.png;
# get_data.sh (Step 3) writes the raw render to data/pages/<page_id>.png. Interim is
# preferred when present, so OCR always sees the cleaned scan the pipeline actually produced.
INTERIM_DIR = Path("data/interim")
PAGES_DIR = Path("data/pages")

# Per-page cache + sidecars. One <page_id>.mmd holds the whole page's raw Nougat markdown
# (also what data/validate.py's word-count floor reads from), meta.jsonl holds one row per
# CHUNK (ocr_confidence + bbox, summary.md 3f), failures.json logs degenerate pages honestly.
OCR_DIR = Path("data/ocr")
META_PATH = OCR_DIR / "meta.jsonl"
FAILURES_PATH = OCR_DIR / "failures.json"

# Nougat's decoder position limit is 4096 tokens (facebook/nougat-base config). We cap well
# under that: a baseline (not yet fine-tuned) reader running to the true limit on a dense
# numeric-table page is exactly the repetition-degeneration failure mode _is_degenerate()
# exists to catch, and paying that wall-clock cost on CPU for a page we are going to discard
# anyway is wasted. 1536 tokens comfortably covers a normal prose/formula page.
MAX_NEW_TOKENS = 1536

# Repetition-degeneration guard (summary.md 3a item 4 / plan.md Step 11 point 9): a stuck
# decoder repeats the same short n-gram forever instead of stopping. Detected as the tail of
# the decoded text decomposing into >=MIN_REPEATS consecutive identical NGRAM-word blocks -- a
# strong, cheap signal that needs no external dependency (the `nougat` package's own stopping
# criterion was ruled out project-wide in summary.md 7a for the same reason: dependency
# conflict with this repo's pinned `transformers`).
DEGEN_NGRAM = 12
DEGEN_MIN_REPEATS = 4

# The citation anchor our Explainable NFR needs (summary.md 3f / 10): A&S formula numbers
# look like "6.1.8". Parsed out of a chunk's OWN text, never guessed.
FORMULA_ID_RE = re.compile(r"\d+\.\d+\.\d+")


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.device = str(cfg.get("device", "cpu"))
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        """Load facebook/nougat-base (or cfg['ocr']['model']) on first use, not at
        construction -- so building a Reader() in a test doesn't force a model download."""
        if self._model is not None:
            return
        import torch
        from transformers import NougatProcessor, VisionEncoderDecoderModel

        model_name = self.cfg.get("model", "facebook/nougat-base")
        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("vision.ocr: cfg requests cuda but no GPU is visible; running on CPU")
            device = "cpu"
        self.device = device

        self._processor = NougatProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        self._model = model

    def _generate(self, image: Any) -> tuple[str, float]:
        """Run one Nougat forward pass on a single image (a full page or a crop).

        Returns (decoded_markdown, confidence). Confidence is the mean per-token
        generation probability (exp of the mean transition log-prob) -- a cheap,
        standard `generate(..., output_scores=True)` readout, not a calibrated metric
        (calibration is the A3 "Calibrated" NFR's job, not this baseline reader's).
        """
        import torch

        self._ensure_loaded()
        pixel_values = self._processor(image, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=MAX_NEW_TOKENS,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                output_scores=True,
                return_dict_in_generate=True,
            )
        sequence = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        sequence = self._processor.post_process_generation(sequence, fix_markdown=False)

        confidence = 0.5  # neutral fallback if transition-score readout is unavailable
        try:
            scores = self._model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            finite = scores[torch.isfinite(scores)]
            if finite.numel() > 0:
                confidence = float(math.exp(float(finite.mean())))
        except Exception:
            logger.warning("vision.ocr: could not compute a confidence score for this page")
        return sequence, confidence

    def transcribe_region(self, region: Region) -> str:
        """Crop -> processor -> model.generate -> decoded LaTeX/markdown string.

        Used for (a) per-region re-OCR when a page fails at the page level, and (b) the
        formula-crop (image, latex) pairs the Sprint-4 fine-tune trains on (plan.md 4b) --
        so this stays real and load-bearing, not a shim kept only to satisfy the locked
        `Reader.transcribe_region` signature.
        """
        from PIL import Image as PILImage

        path = _page_image_path(region.page_id)
        image = PILImage.open(path).convert("RGB").crop(region.bbox)
        text, _confidence = self._generate(image)
        return text


def _page_image_path(page_id: str) -> Path:
    interim = INTERIM_DIR / f"{page_id}.png"
    if interim.exists():
        return interim
    raw = PAGES_DIR / f"{page_id}.png"
    if raw.exists():
        return raw
    raise FileNotFoundError(
        f"vision.ocr: no image for page_id={page_id!r} under {INTERIM_DIR} or {PAGES_DIR}"
    )


def _group_by_page(regions: list[Region]) -> dict[str, list[Region]]:
    """Group regions by page, preserving first-seen page order and each page's own
    region order (both already reading-order, per vision/layout.py)."""
    groups: dict[str, list[Region]] = {}
    for r in regions:
        groups.setdefault(r.page_id, []).append(r)
    return groups


def _is_degenerate(text: str) -> bool:
    """True if the tail of `text` is a stuck repeated n-gram loop (see DEGEN_* above)."""
    tokens = text.split()
    window = DEGEN_NGRAM * DEGEN_MIN_REPEATS
    if len(tokens) < window:
        return False
    tail = tokens[-window:]
    pattern = tail[:DEGEN_NGRAM]
    return all(
        tail[i * DEGEN_NGRAM : (i + 1) * DEGEN_NGRAM] == pattern
        for i in range(1, DEGEN_MIN_REPEATS)
    )


def _split_markdown_to_regions(markdown: str, n_regions: int) -> list[str]:
    """Approximate a page-level Nougat transcript back onto per-region text.

    Nougat is a PAGE-level model (plan.md Step 11 design note) -- it has no notion of our
    layout regions, so there is no exact mapping. We split the markdown on blank lines
    (Nougat already delimits paragraphs/headings/display-equations that way) and pair the
    resulting blocks with this page's regions **in order** -- both sequences are reading
    order, so position-matching is the best available proxy. Extra trailing blocks are
    folded into the last region rather than dropped; a shortfall pads with "" rather than
    raising, so a page is never lost to a block-count mismatch. This heuristic -- and why
    it's an approximation, not an alignment -- is written up in form Section 3/7.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", markdown.strip()) if b.strip()]
    if n_regions <= 0:
        return []
    if not blocks:
        return [""] * n_regions
    if len(blocks) == n_regions:
        return blocks
    if len(blocks) > n_regions:
        head = blocks[: n_regions - 1]
        tail = "\n\n".join(blocks[n_regions - 1 :])
        return head + [tail]
    return blocks + [""] * (n_regions - len(blocks))


def _chunk_id(doc_id: str, page_id: str, region_idx: int, text: str) -> str:
    base = f"{doc_id}|{page_id}|r{region_idx:02d}"
    m = FORMULA_ID_RE.search(text)
    return f"{base}|{m.group(0)}" if m else base


def _load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[row["chunk_id"]] = row
    return out


def _write_jsonl(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row) + "\n")


def _load_failures(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        return {row["page_id"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_failures(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows.values()), indent=2) + "\n", encoding="utf-8")


def transcribe(regions: list[Region], cfg: dict, *, limit_pages: int | None = None) -> list[Chunk]:
    """Regions -> text chunks.

    Groups regions by page and runs Nougat **once per page** (not once per region): Nougat
    was trained on whole pages and uses full-page context, so this is both the efficient and
    the accuracy-preserving path (plan.md Step 11 design note). The page's markdown is then
    approximated back onto that page's regions via _split_markdown_to_regions().

    Resumable: a page whose data/ocr/<page_id>.mmd already exists is not re-run through the
    model -- its cached markdown is re-split against the CURRENT regions instead, so a
    layout.py change still produces up-to-date chunks without paying for inference again
    (plan.md Step 11 point 8; matters because Kaggle sessions die at ~9h, summary.md 11.4).

    `limit_pages` is the "test on N pages without running the whole book" guard (plan.md
    Step 11 point 7): an optional keyword-only cap on how many *distinct pages* worth of
    regions are processed, applied AFTER grouping so a partial page is never split across
    the boundary. It defaults to None (no limit) and is not passed by pipeline.py's fixed
    `ocr.transcribe(regions, cfg)` call, so normal pipeline behaviour is unchanged.

    A page whose decode degenerates into a repeated-token loop (_is_degenerate) is logged to
    data/ocr/failures.json and produces NO chunks -- summary.md 4i: "mark the page as failed
    rather than writing garbage."
    """
    reader = Reader(cfg)
    by_page = _group_by_page(regions)
    page_ids = list(by_page)[:limit_pages] if limit_pages is not None else list(by_page)

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    meta_rows = _load_jsonl(META_PATH)
    failure_rows = _load_failures(FAILURES_PATH)

    chunks: list[Chunk] = []
    n_processed = n_cached = n_failed = 0
    t0 = time.time()

    for page_id in page_ids:
        page_regions = by_page[page_id]
        doc_id = _chapter_of(int(page_id[4:])) if page_id.startswith("as_p") else page_id
        mmd_path = OCR_DIR / f"{page_id}.mmd"

        if mmd_path.exists():
            markdown = mmd_path.read_text(encoding="utf-8")
            # Confidence isn't in the .mmd cache (only the text is); recover it from this
            # page's own prior meta.jsonl rows if a previous run already wrote them.
            prior_confs = [
                row["ocr_conf"]
                for cid, row in meta_rows.items()
                if cid.startswith(f"{doc_id}|{page_id}|")
            ]
            confidence = float(prior_confs[0]) if prior_confs else 0.5
            n_cached += 1
        else:
            image_path = _page_image_path(page_id)
            from PIL import Image as PILImage

            image = PILImage.open(image_path).convert("RGB")
            markdown, confidence = reader._generate(image)

            if _is_degenerate(markdown):
                failure_rows[page_id] = {
                    "page_id": page_id,
                    "reason": "repetition-degeneration",
                    "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _write_failures(FAILURES_PATH, failure_rows)
                n_failed += 1
                logger.warning(f"vision.ocr: {page_id} degenerated (repeated-token loop); skipped")
                continue

            mmd_path.write_text(markdown, encoding="utf-8")
            n_processed += 1

        region_texts = _split_markdown_to_regions(markdown, len(page_regions))
        for idx, (region, text) in enumerate(zip(page_regions, region_texts, strict=True)):
            chunk_id = _chunk_id(doc_id, page_id, idx, text)
            chunks.append(
                Chunk(id=chunk_id, doc_id=doc_id, text=text, page_ids=[page_id], score=0.0)
            )
            meta_rows[chunk_id] = {
                "chunk_id": chunk_id,
                "ocr_conf": round(confidence, 4),
                "bbox": list(region.bbox),
            }

        if (n_processed + n_cached) % 50 == 0 and n_processed:
            rate = n_processed / max(time.time() - t0, 1e-9)
            logger.info(f"vision.ocr: {n_processed} pages transcribed ({rate:.2f}/s)")

    _write_jsonl(META_PATH, meta_rows)
    logger.info(
        f"vision.ocr: {len(chunks)} chunks from {n_processed} newly-transcribed + "
        f"{n_cached} cached pages ({n_failed} failed/degenerate, skipped)"
    )
    return chunks
