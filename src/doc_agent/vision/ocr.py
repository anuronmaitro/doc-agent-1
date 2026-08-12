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

# Step 18b defect 3: generate() set no repetition_penalty, and the spirals in the DEGEN_*
# comments above are exactly what that omission produces. 1.1 is deliberately mild -- A&S
# legitimately repeats subscripts and table rows (a column of "0", a run of "\frac{1}{2}"),
# and HuggingFace's no_repeat_ngram_size would corrupt those outright; a soft per-token
# penalty instead just makes an already-generated token less attractive next time, which
# discourages runaway spirals without forbidding genuine repetition.
REPETITION_PENALTY = 1.1

# Repetition-degeneration guard (summary.md 3a item 4 / plan.md Step 11 point 9): a stuck
# decoder repeats the same short n-gram forever instead of stopping. Detected as the tail of
# the decoded text decomposing into >=MIN_REPEATS consecutive identical NGRAM-word blocks -- a
# strong, cheap signal that needs no external dependency (the `nougat` package's own stopping
# criterion was ruled out project-wide in summary.md 7a for the same reason: dependency
# conflict with this repo's pinned `transformers`).
DEGEN_NGRAM = 12
DEGEN_MIN_REPEATS = 4

# --- three failure modes the tail-only n-gram check above cannot see (found in Step 16) ---
# Measured on Step 16's first Kaggle smoke run: the tail check flagged 1 of 20 pages, while
# 4+ were actually unusable. Each constant below closes one of the gaps that hid them.
#
# 1. Nougat announces its own failures. When it cannot read a page it emits a literal
#    [MISSING_PAGE_POST] / [MISSING_PAGE_EMPTY] / [MISSING_PAGE_FAIL] marker. We were
#    writing those straight to .mmd and counting them as successes -- printed p.243 (a
#    dense table) produced 239 characters consisting of a truncated table header and
#    [MISSING_PAGE_POST], and was reported as a good page.
MISSING_PAGE_RE = re.compile(r"\[MISSING_PAGE[_A-Z]*\]")
#
# 2. A near-empty transcript is a failure, not a short page. Real A&S content pages run
#    to hundreds of characters; the smoke run produced one page of 4 characters and one
#    of 35. The floor sits well under the shortest genuine page observed (239 chars was
#    itself a failure; the shortest sound page was 265).
MIN_PAGE_CHARS = 120
#
# 3. Degeneration ANYWHERE on the page, not just at the tail. The tail check only inspects
#    the last DEGEN_NGRAM * DEGEN_MIN_REPEATS tokens, so a decoder that spirals mid-page and
#    then ends plausibly slips through. Detected as a short character unit repeated many
#    times in a row, which is what these spirals actually look like:
#      - printed p.255 emitted "\!" x603 inside formula 6.1.3, burning the token budget so
#        only 3 of its 14 numbered formulas ever appeared;
#      - printed p.295 read the ch.7 contents list correctly, then ran "<= " to the end.
#    Two weaker signals were measured and REJECTED on the same 19-page sample:
#      - whole-page token diversity: p.255 scored 0.711 unique (threshold would need to be
#        >0.7 to fire) because each "\!\!\!..." run has a different length and so counts as
#        a *distinct* token -- the signal is structurally blind to this failure;
#      - zlib compression ratio: p.255 = 0.183 vs a clean p.065 = 0.224, a margin too thin
#        to set a threshold on without false positives.
#    The repeated-unit count separates cleanly: sound pages topped out at 6 consecutive
#    repeats, the two degenerate pages hit 39 and 38. 20 sits ~3x above the clean maximum
#    and ~2x below the observed failures.
#
#    Step 18b correction: DEGEN_REPEAT_UNIT_MAX_LEN=4 was itself blind to its own dominant
#    failure. Auditing the 594 "successful" Step 16 pages against the PDF's text layer
#    found 41 MORE spiralling pages hiding inside them (91 total; the old detector caught
#    50, i.e. 55%) -- because "\qquad" is 6 characters and "\begin{array}{c}" is 16, both
#    longer than the unit length that could ever match. Widened to 20. That alone would
#    now flag legitimate LaTeX table syntax too -- "c c c c" and "|c|c|c|" are genuine
#    `\begin{tabular}` column specs, not degeneration, and 34 of the original 75 raw hits
#    were exactly this. TABULAR_UNIT_RE excludes any matched unit built ONLY from column-
#    spec characters (alignment letters, bars, braces, digits, whitespace, and "&", the
#    cell separator -- p.328's flagged unit was a bare "&" from a sparse table row, not a
#    spiral) -- a real spiral is always a backslash macro or math content, never just that.
#
#    The repeat count itself was re-checked against Elias's 11 known-genuine spirals and
#    dropped from 20 to 13, for two independent reasons:
#      - exact-match fragility: real spirals decode with a stray whitespace inserted every
#        ~13-14 copies (e.g. "\qquad\qquad...\qquad \qquad..."), which breaks a strict
#        backreference at 20 copies outright -- p.360 and p.177 were both missed this way,
#        p.360 being the exact gold page this repair exists to fix. Matched against the
#        text with ALL whitespace stripped first (LaTeX macros are whitespace-insensitive;
#        a decoder stuck on a token is stuck regardless of incidental spacing), not the
#        word-tokenized `stripped` used by the tail check below.
#      - p.289's genuine spiral only repeats its unit 13 times total, never reaching 20.
#    13 is the lowest threshold that still catches all 11 known cases. Lowering it further
#    starts catching short units (e.g. "\," x14 = ~1% of an otherwise-good page) that read
#    as coincidental formula spacing rather than a stuck decoder, so a MIN_SPIRAL_SPAN_CHARS
#    floor (naturally scaling with unit length) guards against exactly that.
# Step 28 correction (2026-08-12): widened 20 -> 60 after the fine-tuned reader's real
# Kaggle validation run produced a spiral this threshold still missed. `as_p0334`'s
# lowest-scoring prediction (char-F1 0.067, curve point n=122) repeats the unit
# `-\mu xP_{\tau}^{n}(z) ` -- 22 characters, past the old 20-char cap -- more than a dozen
# times, and was scored as a low-quality "success" instead of counted as a failure because
# the detector's own unit-length window couldn't see it. Found by actually reading the
# generated text, not just the aggregate char-F1 number, the same discipline that found
# the original DEGEN_REPEAT_UNIT_MAX_LEN=4 -> 20 gap at Step 18b. 60 gives real headroom
# above the one measured case rather than being set to exactly fit it.
DEGEN_REPEAT_UNIT_MAX_LEN = 60
DEGEN_MIN_UNIT_REPEATS = 13
# Compiles to (.{1,60}?)\1{12,} : a 1-60 character unit, then 12 more copies = 13 total.
DEGEN_REPEAT_RE = re.compile(
    rf"(.{{1,{DEGEN_REPEAT_UNIT_MAX_LEN}}}?)\1{{{DEGEN_MIN_UNIT_REPEATS - 1},}}",
    re.DOTALL,
)
TABULAR_UNIT_RE = re.compile(r"^[lcr|@{}&\s.0-9]*$")
WS_RE = re.compile(r"\s+")
MIN_SPIRAL_SPAN_CHARS = 60

# Step 21 finding 1 (block-level repetition), implemented at Step 28: a WHOLE block
# (paragraph or display equation, separated from its neighbors by a blank line) repeating
# verbatim later in the same page is a different failure shape from DEGEN_REPEAT_RE above
# -- that regex looks for a short-to-medium unit repeating CONSECUTIVELY, not one block
# reappearing once, much later, with different content in between. Measured on the 20
# validation pages: `as_p0340` emits 3 blocks twice (19% of the page duplicated),
# `as_p0441` emits 2 display equations twice (14%) -- both recorded as successes by the
# unit-regex check alone. Exact-match only (not near-duplicate/fuzzy): both measured cases
# are byte-identical repeats, and exact match is the check least likely to false-positive
# on legitimate content that merely looks similar (e.g. two different rows of a table that
# happen to share most of their text).
MIN_BLOCK_DUP_CHARS = 60
_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")


def _has_duplicate_block(text: str) -> bool:
    """True if any block (paragraph/equation, split on blank lines) of at least
    `MIN_BLOCK_DUP_CHARS` characters appears more than once, verbatim, in `text`."""
    seen: set[str] = set()
    for block in _BLOCK_SPLIT_RE.split(text):
        block = block.strip()
        if len(block) < MIN_BLOCK_DUP_CHARS:
            continue
        if block in seen:
            return True
        seen.add(block)
    return False


# Step 18b defect 5, found on the full-book run (not the 20-page smoke sample): a region
# crop with a near-zero width or height crashes Nougat's OWN preprocessing, not ours.
# layout.detect() (TATR, a learned model) does not guarantee a sane bbox on every region --
# one page produced a crop of shape (1, 1325, 3). HF's image_processing_nougat.crop_margin()
# calls to_channel_dimension_format() on that array; its "channel dim is ambiguous" heuristic
# reads a leading size-1 axis as channels-first, and the resulting transpose((2,0,1)) raises
# `ValueError: axes don't match array` -- an uncaught exception that took the entire ~5h
# Kaggle run down with it (papermill has no per-cell recovery). There is no content to read
# in a 1-pixel-tall sliver anyway, so skip the model call rather than let it reach the crash.
MIN_CROP_DIM_PX = 8

# The citation anchor our Explainable NFR needs (summary.md 3f / 10): A&S formula numbers
# look like "6.1.8". Parsed out of a chunk's OWN text, never guessed.
FORMULA_ID_RE = re.compile(r"\d+\.\d+\.\d+")

# Pinned commit for cfg["ocr"]["model"]'s locked default (facebook/nougat-base) -- bandit
# B615 flags from_pretrained() without a revision as a supply-chain risk, since an unpinned
# model name can resolve to different weights later. Resolved from that repo's `main` ref
# at implementation time; bump deliberately, not implicitly, if it ever needs to move.
NOUGAT_REVISION = "abfecedbb34367c820e233f710fdc7f54e6ab249"


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.device = str(cfg.get("device", "cpu"))
        self._model: Any = None
        self._processor: Any = None
        self._dtype: Any = None  # resolved in _ensure_loaded (fp16 on GPU, fp32 on CPU)

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

        # Pinned commit for the locked default (bandit B615: an unpinned model name can
        # resolve to different weights later -- same fix vision/layout.py already applies
        # to its own from_pretrained() call, for the same reason). A differently configured
        # model name (not something this project's config.yaml allows) falls back to
        # unpinned, matching from_pretrained's own default resolution.
        revision = NOUGAT_REVISION if model_name == "facebook/nougat-base" else None

        # Half precision on GPU (Step 16). Nougat's own reference implementation runs
        # fp16, and autoregressive decoding is the dominant cost of a full-book pass:
        # Step 16's first Kaggle run measured 17.4 s/page in fp32 on a T4, i.e. ~5.5 h
        # for the 1040-page corpus. CPU stays fp32 -- half precision there is slower,
        # not faster, and unsupported for some ops.
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._dtype = dtype

        self._processor = NougatProcessor.from_pretrained(model_name, revision=revision)
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name, revision=revision, torch_dtype=dtype
        )
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
        # Pixel values must match the model's dtype -- fp16 weights with fp32 inputs
        # raises rather than silently upcasting.
        pixel_values = self._processor(image, return_tensors="pt").pixel_values.to(
            self.device, dtype=self._dtype
        )
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=MAX_NEW_TOKENS,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                repetition_penalty=REPETITION_PENALTY,
                output_scores=True,
                return_dict_in_generate=True,
            )
        sequence = self._processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        sequence = self._processor.post_process_generation(sequence, fix_markdown=False)

        confidence = 0.5  # neutral fallback if the score readout is unavailable
        try:
            # Score the chosen tokens directly instead of calling
            # model.compute_transition_scores(..., normalize_logits=True). That helper
            # reshapes by `self.config.vocab_size`, which a VisionEncoderDecoderConfig
            # does not define -- the decoder's vocabulary lives at
            # config.decoder.vocab_size (50000 for nougat-base). It therefore raised
            # AttributeError on EVERY page and the bare except left confidence pinned at
            # the 0.5 fallback: Step 16's first Kaggle run wrote 201 chunk rows whose
            # ocr_conf was identically 0.5, a constant masquerading as a measurement.
            # Doing the log-softmax ourselves is both correct and version-proof.
            if outputs.scores:
                step_logits = torch.stack(outputs.scores, dim=1)[0].float()  # (steps, vocab)
                gen_ids = outputs.sequences[0, -step_logits.shape[0] :]
                logprobs = torch.log_softmax(step_logits, dim=-1)
                chosen = logprobs[torch.arange(gen_ids.shape[0], device=logprobs.device), gen_ids]
                finite = chosen[torch.isfinite(chosen)]
                if finite.numel() > 0:
                    confidence = float(math.exp(float(finite.mean())))
        except Exception as exc:
            # Log the actual exception. The previous version swallowed it, which is why
            # a per-page failure went unnoticed for an entire GPU run.
            logger.warning(
                f"vision.ocr: confidence unavailable for this page "
                f"({type(exc).__name__}: {exc})"
            )
        return sequence, confidence

    def _generate_region(self, region: Region) -> tuple[str, float]:
        """Crop -> processor -> model.generate -> (decoded text, confidence).

        The shared implementation behind `transcribe_region` (below) and Step 18b defect
        1's page-level retry in `transcribe()`, which needs the confidence value that
        `transcribe_region`'s locked `-> str` signature has nowhere to return.
        """
        from PIL import Image as PILImage

        path = _page_image_path(region.page_id)
        image = PILImage.open(path).convert("RGB").crop(region.bbox)
        if image.width < MIN_CROP_DIM_PX or image.height < MIN_CROP_DIM_PX:
            logger.warning(
                f"vision.ocr: {region.page_id} region bbox={region.bbox} crops to "
                f"{image.width}x{image.height}px (degenerate); skipping the model call"
            )
            return "", 0.0
        return self._generate(image)

    def transcribe_region(self, region: Region) -> str:
        """Crop -> processor -> model.generate -> decoded LaTeX/markdown string.

        Used for (a) per-region re-OCR when a page fails at the page level, and (b) the
        formula-crop (image, latex) pairs the Sprint-4 fine-tune trains on (plan.md 4b) --
        so this stays real and load-bearing, not a shim kept only to satisfy the locked
        `Reader.transcribe_region` signature.
        """
        text, _confidence = self._generate_region(region)
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


def _failure_reason(text: str) -> str | None:
    """Why this page's transcript is unusable, or None if it looks sound.

    Returns a short machine-readable reason so data/ocr/failures.json records *how* a
    page failed, not merely that it did -- form Section 5 asks us to report failures
    honestly, and "20% of pages failed, here is the breakdown by mode" is a far more
    useful admission than a bare count. Ordered cheapest check first.
    """
    if MISSING_PAGE_RE.search(text):
        return "nougat-missing-page-marker"

    stripped = text.strip()
    if len(stripped) < MIN_PAGE_CHARS:
        return "empty-or-near-empty"

    # Whole-page repetition (catches mid-page spirals the tail check misses). Matched
    # against the whitespace-collapsed text (see DEGEN_MIN_UNIT_REPEATS above) so a stray
    # space every ~13-14 copies can't break the backreference. Scans every match, not just
    # the first: a page can open with a legitimate tabular block and still spiral later, so
    # stopping at the first hit would let that page through. A MIN_SPIRAL_SPAN_CHARS floor
    # keeps short-unit coincidental repeats (formula spacing like "\," or "\!") from firing
    # on a handful of copies that only cover a sliver of an otherwise-good page.
    no_ws = WS_RE.sub("", stripped)
    for m in DEGEN_REPEAT_RE.finditer(no_ws):
        span = m.end() - m.start()
        if span >= MIN_SPIRAL_SPAN_CHARS and not TABULAR_UNIT_RE.match(m.group(1)):
            return "repetition-degeneration"

    # Block-level repetition (Step 21 finding 1, implemented at Step 28): a whole
    # paragraph/equation block repeating once, verbatim, much later in the page -- a
    # different shape from the consecutive-unit spiral above, so it needs its own check
    # rather than a bigger DEGEN_REPEAT_UNIT_MAX_LEN. See `_has_duplicate_block`'s
    # docstring for the two real pages (as_p0340, as_p0441) that motivated this.
    if _has_duplicate_block(stripped):
        return "block-repetition-degeneration"

    tokens = stripped.split()

    # Original tail check: a decoder still looping when generation was cut off.
    window = DEGEN_NGRAM * DEGEN_MIN_REPEATS
    if len(tokens) >= window:
        tail = tokens[-window:]
        pattern = tail[:DEGEN_NGRAM]
        if all(
            tail[i * DEGEN_NGRAM : (i + 1) * DEGEN_NGRAM] == pattern
            for i in range(1, DEGEN_MIN_REPEATS)
        ):
            return "repetition-degeneration"

    return None


def _is_degenerate(text: str) -> bool:
    """True if this page's transcript is unusable for any reason (see _failure_reason)."""
    return _failure_reason(text) is not None


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


def _retry_page_by_region(
    reader: Reader, page_regions: list[Region]
) -> tuple[str, list[str], list[float]] | None:
    """Step 18b defect 1 fix: when whole-page generation fails, retry region-by-region
    instead of discarding the page untried. `Reader.transcribe_region`'s own docstring
    already says it exists for exactly this ("per-region re-OCR when a page fails") --
    Step 16 never actually called it, so every failed page was thrown away regardless.

    A single-column region crop is much closer to Nougat's training distribution (modern
    single-column arXiv papers) than a two-column 1964 scan, which is precisely the
    layout defect 4 (early stopping) is measured against -- so this is a real second
    chance, not a formality.

    Returns `(recombined_markdown, region_texts, region_confidences)` -- one text and one
    confidence PER REGION, in reading order, so downstream chunk-building can use the real
    per-region confidence instead of a single page-level scalar -- or `None` if the retry
    is *also* unusable, in which case the caller keeps the original failure.
    """
    region_texts: list[str] = []
    region_confs: list[float] = []
    for region in page_regions:
        # This loop is defect 1's own fix, exercised for the first time at full-book scale
        # in Step 18b -- and it found a crash (defect 5: a degenerate crop dimension, guarded
        # in _generate_region above) that took an entire ~5h unattended run down with it. The
        # per-region guard fixes the KNOWN cause; this except is the belt-and-suspenders for
        # an unknown one -- one bad region among ~1040 pages' worth must not cost the whole
        # job again. Treated the same as a genuinely blank region: empty text, zero confidence.
        try:
            text, conf = reader._generate_region(region)
        except Exception as exc:
            logger.warning(
                f"vision.ocr: region retry crashed on {region.page_id} bbox={region.bbox} "
                f"({type(exc).__name__}: {exc}); treating as empty"
            )
            text, conf = "", 0.0
        region_texts.append(text)
        region_confs.append(conf)
    recombined = "\n\n".join(t for t in region_texts if t.strip())
    if _failure_reason(recombined) is not None:
        return None
    return recombined, region_texts, region_confs


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

    A page whose whole-page decode fails (_failure_reason) is not discarded immediately --
    Step 18b defect 1: it gets ONE region-by-region retry (_retry_page_by_region) before
    being logged to data/ocr/failures.json and producing NO chunks. Only a page that fails
    *both* the whole-page attempt and the region retry is actually given up on -- summary.md
    4i: "mark the page as failed rather than writing garbage" still holds, it just now
    happens after a real second chance, not on the first bad decode.
    """
    reader = Reader(cfg)
    by_page = _group_by_page(regions)
    page_ids = list(by_page)[:limit_pages] if limit_pages is not None else list(by_page)

    OCR_DIR.mkdir(parents=True, exist_ok=True)
    meta_rows = _load_jsonl(META_PATH)
    failure_rows = _load_failures(FAILURES_PATH)

    chunks: list[Chunk] = []
    n_processed = n_cached = n_failed = n_recovered = 0
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
            region_texts = _split_markdown_to_regions(markdown, len(page_regions))
            region_confs = [confidence] * len(page_regions)
            n_cached += 1
        else:
            image_path = _page_image_path(page_id)
            from PIL import Image as PILImage

            image = PILImage.open(image_path).convert("RGB")
            markdown, confidence = reader._generate(image)

            reason = _failure_reason(markdown)
            if reason is not None:
                # Defect 1 fix: don't discard the page untried -- retry region-by-region
                # before giving up. A single-column crop is closer to Nougat's training
                # distribution than the two-column 1964 scan that just failed whole.
                retry = _retry_page_by_region(reader, page_regions)
                if retry is None:
                    failure_rows[page_id] = {
                        "page_id": page_id,
                        "reason": reason,
                        "chars": len(markdown.strip()),
                        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    _write_failures(FAILURES_PATH, failure_rows)
                    n_failed += 1
                    logger.warning(
                        f"vision.ocr: {page_id} unusable ({reason}); region retry also failed; skipped"
                    )
                    continue
                markdown, region_texts, region_confs = retry
                n_recovered += 1
                logger.info(
                    f"vision.ocr: {page_id} recovered via region-level retry "
                    f"(page-level attempt: {reason})"
                )
            else:
                region_texts = _split_markdown_to_regions(markdown, len(page_regions))
                region_confs = [confidence] * len(page_regions)

            mmd_path.write_text(markdown, encoding="utf-8")
            n_processed += 1

        for idx, (region, text, conf) in enumerate(
            zip(page_regions, region_texts, region_confs, strict=True)
        ):
            chunk_id = _chunk_id(doc_id, page_id, idx, text)
            chunks.append(
                Chunk(id=chunk_id, doc_id=doc_id, text=text, page_ids=[page_id], score=0.0)
            )
            meta_rows[chunk_id] = {
                "chunk_id": chunk_id,
                "ocr_conf": round(conf, 4),
                "bbox": list(region.bbox),
            }

        if (n_processed + n_cached) % 50 == 0 and n_processed:
            rate = n_processed / max(time.time() - t0, 1e-9)
            logger.info(f"vision.ocr: {n_processed} pages transcribed ({rate:.2f}/s)")

    _write_jsonl(META_PATH, meta_rows)
    logger.info(
        f"vision.ocr: {len(chunks)} chunks from {n_processed} newly-transcribed "
        f"({n_recovered} via region-level retry) + {n_cached} cached pages "
        f"({n_failed} failed/degenerate, skipped)"
    )
    return chunks
