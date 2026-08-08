"""Stage 1 — load scanned page-images"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..contracts import *  # noqa

# Fixed on-disk convention set by scripts/get_data.sh (not part of cfg): printed content
# pages are data/pages/as_pNNNN.png, front matter is as_fNNNN.png (see data/provenance.md).
PAGES_DIR = Path("data/pages")

# First PRINTED page of each chapter — mirrors notebooks/eda.ipynb cell 8 (chapter_of()),
# which data/validate.py's ALL_CHAPTERS key set also mirrors. doc_id IS the chapter id
# (summary.md 3f), so this map is the single source every downstream doc_id is derived from.
CHAPTER_STARTS: dict[str, int] = {
    "ch01_math_constants": 1,
    "ch02_phys_constants": 5,
    "ch03_elem_analytic": 9,
    "ch04_elem_transcend": 65,
    "ch05_expint": 227,
    "ch06_gamma": 253,
    "ch07_error_fresnel": 295,
    "ch08_legendre": 331,
    "ch09_bessel": 355,
    "ch10_bessel_frac": 435,
    "ch11_bessel_integrals": 479,
    "ch12_struve": 495,
    "ch13_conf_hypergeom": 503,
    "ch14_coulomb": 537,
    "ch15_hypergeom": 555,
    "ch16_jacobi_elliptic": 567,
    "ch17_elliptic_integrals": 587,
    "ch18_weierstrass": 627,
    "ch19_parabolic_cyl": 685,
    "ch20_mathieu": 721,
    "ch21_spheroidal": 751,
    "ch22_orthogonal_poly": 771,
    "ch23_bernoulli_zeta": 803,
    "ch24_combinatorial": 821,
    "ch25_num_interp": 875,
    "ch26_probability": 925,
    "ch27_misc": 997,
    "ch28_scales_notation": 1011,
    "ch29_laplace": 1019,
}
_CHAPTER_ORDER = sorted(CHAPTER_STARTS.items(), key=lambda kv: kv[1])

# A blank verso / pure-plate page carries almost no ink variation. Measured over all
# 1050 as_p* pages in the real corpus: the 10 lowest-contrast pages cluster at
# grayscale std 5.3-8.8, then jump to 10.9+ for the first genuine (if sparse) content
# page -- a clean natural gap in the measured data, not an assumed cutoff.
BLANK_STD_THRESHOLD = 10.0


def _chapter_of(printed_page: int) -> str:
    """Map a printed page number to its chapter id (= Chunk.doc_id)."""
    chapter = _CHAPTER_ORDER[0][0]
    for name, start in _CHAPTER_ORDER:
        if printed_page >= start:
            chapter = name
        else:
            break
    return chapter


def _is_content_page(path: Path) -> bool:
    """Drop blank versos / pure-plate pages: real content has visible ink variation."""
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return bool(arr.std() >= BLANK_STD_THRESHOLD)


def load_pages(cfg: dict) -> list[Page]:
    """Read data/pages/*.png -> list[Page], sorted by printed page number.

    Only as_pNNNN.png (printed content pages) are loaded; as_fNNNN.png front-matter
    pages and blank-verso/pure-plate pages (see _is_content_page) are dropped, per
    summary.md 3d / data/provenance.md, leaving ~1046 of the 1050 printed pages.
    Page.doc_id is the chapter id computed from the printed page number (summary.md 3f).
    """
    pages: list[Page] = []
    for f in sorted(PAGES_DIR.glob("as_p*.png")):
        printed = int(f.stem[4:])
        if not _is_content_page(f):
            continue
        pages.append(
            Page(
                id=f.stem,
                image_path=str(f).replace("\\", "/"),
                doc_id=_chapter_of(printed),
            )
        )
    pages.sort(key=lambda p: int(p.id[4:]))
    return pages
