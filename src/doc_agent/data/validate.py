"""Data — data schema/quality validation at ingest"""

from __future__ import annotations

from pathlib import Path

from ..config import load_task
from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# --- the canonical 29-chapter split (mirrors notebooks/eda.ipynb, cells 8 & 10) -----------
# doc_id IS the chapter id (summary.md 3f): making the chapter the "document" is what lets
# this module assert "split by document, not page" directly from Page.doc_id alone.
TEST_CHAPTERS = frozenset(
    {
        "ch05_expint",
        "ch06_gamma",
        "ch07_error_fresnel",
        "ch09_bessel",
        "ch17_elliptic_integrals",
    }
)
VAL_CHAPTERS = frozenset(
    {
        "ch08_legendre",
        "ch10_bessel_frac",
        "ch13_conf_hypergeom",
        "ch15_hypergeom",
    }
)
ALL_CHAPTERS = frozenset(
    {
        "ch01_math_constants",
        "ch02_phys_constants",
        "ch03_elem_analytic",
        "ch04_elem_transcend",
        "ch05_expint",
        "ch06_gamma",
        "ch07_error_fresnel",
        "ch08_legendre",
        "ch09_bessel",
        "ch10_bessel_frac",
        "ch11_bessel_integrals",
        "ch12_struve",
        "ch13_conf_hypergeom",
        "ch14_coulomb",
        "ch15_hypergeom",
        "ch16_jacobi_elliptic",
        "ch17_elliptic_integrals",
        "ch18_weierstrass",
        "ch19_parabolic_cyl",
        "ch20_mathieu",
        "ch21_spheroidal",
        "ch22_orthogonal_poly",
        "ch23_bernoulli_zeta",
        "ch24_combinatorial",
        "ch25_num_interp",
        "ch26_probability",
        "ch27_misc",
        "ch28_scales_notation",
        "ch29_laplace",
    }
)
BUILD_CHAPTERS = ALL_CHAPTERS - VAL_CHAPTERS - TEST_CHAPTERS

assert len(ALL_CHAPTERS) == 29, "A&S has 29 chapters"
assert BUILD_CHAPTERS.isdisjoint(VAL_CHAPTERS), "LEAK: build and val share a chapter"
assert BUILD_CHAPTERS.isdisjoint(TEST_CHAPTERS), "LEAK: build and test share a chapter"
assert VAL_CHAPTERS.isdisjoint(TEST_CHAPTERS), "LEAK: val and test share a chapter"

# --- the 164 hand-annotated pages (Step 18) -----------------------------------------------
# Printed page numbers, copied from summary.md 4h, which is the source of truth. They live
# here rather than in scripts/get_data.sh so that both the renderer and the test suite read
# ONE list: a page list that exists twice is a page list that will disagree with itself.
#
# summary.md 4h asserts the three sets are "disjoint by construction -- they come from
# disjoint chapter sets". `validate_annotation_sets()` below turns that sentence into a
# checked claim: a stray page from a TEST chapter in the training set is exactly the kind of
# leak that inflates the Step 29 number and cannot be spotted by eye in 164 integers.
ANNOT_TRAIN_PAGES: tuple[int, ...] = (
    1, 2, 3, 7, 15, 24, 40, 44, 50, 51, 70, 75, 114, 115, 140, 141,
    480, 481, 482, 483, 486, 487, 497, 498, 499,
    540, 541, 542, 548, 551, 552, 571, 572, 575, 579, 580,
    631, 633, 634, 639, 640, 659, 693, 702, 703, 707, 709, 711,
    724, 726, 737, 742, 748, 749, 754, 765, 766, 767, 768, 769,
    777, 778, 781, 782, 783, 785, 804, 805, 806, 809, 813, 818,
    823, 852, 858, 862, 864, 865, 878, 879, 883, 885, 886, 915,
    932, 934, 937, 945, 946, 974, 998, 1000, 1004, 1007, 1008, 1010,
    1013, 1014, 1017, 1022, 1023, 1025, 1026, 1027, 1028,
)  # fmt: skip
ANNOT_VAL_PAGES: tuple[int, ...] = (
    332, 334, 340, 350, 351,
    438, 439, 441, 444, 453, 459,
    505, 507, 509, 518, 528, 534,
    562, 563, 564,
)  # fmt: skip
ANNOT_TEST_PAGES: tuple[int, ...] = (
    229, 230, 232, 234, 242, 243, 247,
    255, 256, 258, 259, 260, 262, 280, 281,
    298, 301, 302, 303, 312, 318, 324,
    360, 361, 362, 366, 367, 368, 369, 376, 379, 383, 385, 423,
    600, 619, 621, 622, 625,
)  # fmt: skip

# split name -> (pages, the chapter family every one of those pages must belong to)
ANNOT_SETS: dict[str, tuple[tuple[int, ...], frozenset[str]]] = {
    "train": (ANNOT_TRAIN_PAGES, BUILD_CHAPTERS),
    "val": (ANNOT_VAL_PAGES, VAL_CHAPTERS),
    "test": (ANNOT_TEST_PAGES, TEST_CHAPTERS),
}
ANNOT_EXPECTED_COUNTS = {"train": 105, "val": 20, "test": 39}

# Already transcribed during A1 and reused verbatim (summary.md 4h) -- 36 of the 39 test
# pages still need doing, not 39.
ANNOT_TEST_ALREADY_DONE: frozenset[int] = frozenset({243, 255, 360})


def validate_annotation_sets() -> dict[str, int]:
    """Check the three annotation page lists before anyone spends a day transcribing.

    Verifies, in order:
      1. each list has the size summary.md 4h promises (105 / 20 / 39 = 164),
      2. no page is repeated inside a list,
      3. the three lists are pairwise disjoint,
      4. every page maps to a chapter in its own split's chapter family -- the property
         that makes "no notation leaks between splits" true rather than merely intended.

    Returns the per-split counts. Raises ValueError naming the offending pages otherwise.
    """
    from ..ingest.loader import _chapter_of  # local import: avoids an ingest<-data cycle

    for name, (pages, _family) in ANNOT_SETS.items():
        expected = ANNOT_EXPECTED_COUNTS[name]
        if len(pages) != expected:
            raise ValueError(
                f"validate_annotation_sets: {name} has {len(pages)} pages, expected {expected} "
                "(summary.md 4h)"
            )
        duplicates = sorted({p for p in pages if pages.count(p) > 1})
        if duplicates:
            raise ValueError(f"validate_annotation_sets: {name} repeats page(s) {duplicates}")

    names = list(ANNOT_SETS)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = sorted(set(ANNOT_SETS[a][0]) & set(ANNOT_SETS[b][0]))
            if shared:
                raise ValueError(
                    f"validate_annotation_sets: LEAK — {a} and {b} share page(s) {shared}"
                )

    for name, (pages, family) in ANNOT_SETS.items():
        wrong = {p: _chapter_of(p) for p in pages if _chapter_of(p) not in family}
        if wrong:
            raise ValueError(
                f"validate_annotation_sets: LEAK — {len(wrong)} {name} page(s) fall outside "
                f"the {name} chapter family: {dict(sorted(wrong.items())[:5])}"
                + (" ..." if len(wrong) > 5 else "")
            )

    counts = {name: len(pages) for name, (pages, _f) in ANNOT_SETS.items()}
    logger.info(
        f"validate_annotation_sets: OK — {sum(counts.values())} pages "
        f"({counts['train']} train / {counts['val']} val / {counts['test']} test), "
        "pairwise disjoint, each inside its own chapter family"
    )
    return counts


def _words_from_our_ocr(pages: list[Page], ocr_dir: Path) -> tuple[int, int]:
    """Sum words from OUR OWN OCR output, never the archive's bundled text layer.

    Convention: one `data/ocr/<page.id>.mmd` file per page (vision/ocr.py writes these).
    Returns (total_words, pages_with_ocr) so a shortfall is diagnosable, not just a number.
    """
    total = 0
    covered = 0
    for p in pages:
        f = ocr_dir / f"{p.id}.mmd"
        if f.exists():
            total += len(f.read_text(encoding="utf-8").split())
            covered += 1
    return total, covered


def validate(
    pages: list[Page],
    *,
    ocr_dir: str | Path = "data/ocr",
    task_path: str | Path = "configs/task.yaml",
) -> None:
    """Assert min pages/words, format, no leakage across splits.

    - Page schema: every element is a `Page` with non-empty id/image_path/doc_id, and no
      page id is claimed by two different doc_ids (a loader bug, not a data problem).
    - Corpus size: len(pages) and word count against `configs/task.yaml`'s corpus floor.
    - Word count comes from OUR OWN OCR output in `ocr_dir` (see `_words_from_our_ocr`),
      never the archive's text layer (summary.md watch-outs: "must come from OUR OCR").
    - Split leakage: every doc_id is one of the 29 known chapters, and the fixed
      BUILD/VAL/TEST chapter partition is re-checked disjoint on every call (catches a
      corrupted edit to this file, not just a corrupted corpus).

    Raises ValueError/TypeError on any violation; returns None (logs a summary) if clean.

    `ocr_dir` / `task_path` are keyword overrides for tests; real callers use the
    defaults, i.e. plain `validate(pages)`.
    """
    if not pages:
        raise ValueError("validate: got an empty page list")

    seen_doc_id: dict[str, str] = {}
    for p in pages:
        if not isinstance(p, Page):
            raise TypeError(f"validate: expected Page, got {type(p).__name__}")
        if not p.id or not p.image_path or not p.doc_id:
            raise ValueError(f"validate: Page has an empty required field: {p!r}")
        prior = seen_doc_id.get(p.id)
        if prior is not None and prior != p.doc_id:
            raise ValueError(
                f"validate: page id {p.id!r} maps to two different doc_ids "
                f"({prior!r} and {p.doc_id!r}) — loader bug"
            )
        seen_doc_id[p.id] = p.doc_id

    task = load_task(task_path)
    min_pages = int(task["corpus"]["min_pages"])
    min_words = int(task["corpus"]["min_words"])

    n_pages = len(seen_doc_id)  # unique page ids
    if n_pages < min_pages:
        raise ValueError(f"validate: {n_pages} pages < required minimum {min_pages}")

    n_words, n_covered = _words_from_our_ocr(pages, Path(ocr_dir))
    if n_words < min_words:
        raise ValueError(
            f"validate: {n_words} words from our OCR ({n_covered}/{n_pages} pages have "
            f"{ocr_dir} output) < required minimum {min_words}. "
            "Run vision.ocr.transcribe() over the corpus first (Sprint 2/3), "
            "then re-validate — the floor must be met by our OCR, not the archive text layer."
        )

    unknown = {doc_id for doc_id in seen_doc_id.values() if doc_id not in ALL_CHAPTERS}
    if unknown:
        raise ValueError(
            f"validate: unknown chapter doc_id(s), not in the 29-chapter map: {sorted(unknown)}"
        )

    # Re-assert the partition itself on every call, not just once at import time — a
    # future edit to this file that breaks disjointness must fail loudly here too.
    assert BUILD_CHAPTERS.isdisjoint(VAL_CHAPTERS), "LEAK: build and val share a chapter"
    assert BUILD_CHAPTERS.isdisjoint(TEST_CHAPTERS), "LEAK: build and test share a chapter"
    assert VAL_CHAPTERS.isdisjoint(TEST_CHAPTERS), "LEAK: val and test share a chapter"
    assert BUILD_CHAPTERS | VAL_CHAPTERS | TEST_CHAPTERS == ALL_CHAPTERS, "a chapter is unassigned"

    n_build = sum(1 for d in seen_doc_id.values() if d in BUILD_CHAPTERS)
    n_val = sum(1 for d in seen_doc_id.values() if d in VAL_CHAPTERS)
    n_test = sum(1 for d in seen_doc_id.values() if d in TEST_CHAPTERS)

    logger.info(
        f"validate: OK — {n_pages} pages ({n_build} build / {n_val} val / {n_test} test "
        f"chapters), {n_words} words from {n_covered}/{n_pages} OCR'd pages"
    )
