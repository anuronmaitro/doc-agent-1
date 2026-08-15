"""Step 30 — re-OCR the full ~1046-page book with the fine-tuned reader (curve_n122 +
region-routing hybrid, Step 28 point 11) and rebuild the index.

Runs the same pipeline.build_knowledge_base() stages `scripts/build_index.sh` / `make
ingest index` do, but as a direct Python driver instead of a bash+Makefile indirection, so
`--smoke`/`--limit-pages` can pass through to `ocr.transcribe()` (`build_index.sh` has no
way to do this) -- needed for the timing-projection pre-flight plan.md's own Step 30 text
requires before committing to the real multi-hour run.

Resumable by construction, same guarantee `ocr.transcribe()` already provides: a page
whose data/ocr/<page_id>.mmd exists is not re-run through the model. A killed/timed-out
Kaggle session's partial data/ocr/ + data/interim/ + data/index/embed_cache.npz, re-seeded
into a fresh clone (see this notebook's own "Resuming after a timeout" section), picks up
exactly where it left off.

Usage (Kaggle, GPU, from the repo root):
    python KAGGLE/step30_reindex/run_step30.py           # full ~1046-page run
    python KAGGLE/step30_reindex/run_step30.py --smoke    # tiny N-page timing/correctness check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from doc_agent import config, hooks, wiring  # noqa: E402
from doc_agent.data.validate import validate  # noqa: E402
from doc_agent.index import chunk, embed, store  # noqa: E402
from doc_agent.ingest import enhance, loader, preprocess  # noqa: E402
from doc_agent.logging_conf import get_logger  # noqa: E402
from doc_agent.vision import layout, ocr  # noqa: E402
from doc_agent.vision.ocr import OCR_DIR  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="limit to a small page count for a fast timing/correctness check",
    )
    p.add_argument(
        "--limit-pages",
        type=int,
        default=None,
        help="explicit page cap (overrides --smoke's default of 5 if both given)",
    )
    p.add_argument(
        "--skip-known-failures",
        action="store_true",
        help=(
            "don't re-run pages already in data/ocr/failures.json. Only correct when "
            "resuming with the SAME reader (decoding is greedy, so the retry is "
            "deterministic and cannot succeed). Saved 2.10h on Step 30's index-rebuild "
            "push, where all 53 failures were already recorded."
        ),
    )
    args = p.parse_args()

    limit_pages = args.limit_pages if args.limit_pages is not None else (5 if args.smoke else None)

    cfg = config.load()
    logger.info(
        f"run_step30: ocr cfg -- finetune={cfg['ocr'].get('finetune')} "
        f"adapter_dir={cfg['ocr'].get('adapter_dir')} "
        f"table_adapter_dir={cfg['ocr'].get('table_adapter_dir')} "
        f"limit_pages={limit_pages} skip_known_failures={args.skip_known_failures}"
    )

    wiring.register_all(cfg)

    t0 = time.time()
    pages = loader.load_pages(cfg)
    logger.info(f"run_step30: {len(pages)} pages loaded from data/pages/")
    if limit_pages is not None:
        pages = pages[:limit_pages]
        logger.info(f"run_step30: --smoke/--limit-pages active, capped to {len(pages)} pages")

    pages = preprocess.run(pages, cfg)
    pages = enhance.run(pages, cfg)  # no-op by design (enhance.enabled=false, A1 trade-off)
    hooks.run(hooks.AFTER_INGEST, {"pages": pages})

    t_layout = time.time()
    regions = layout.detect(pages, cfg)
    logger.info(
        f"run_step30: layout.detect() found {len(regions)} regions across {len(pages)} pages ({time.time() - t_layout:.1f}s)"
    )

    t_ocr = time.time()
    chunks = ocr.transcribe(
        regions, cfg, limit_pages=limit_pages, skip_known_failures=args.skip_known_failures
    )
    ocr_elapsed = time.time() - t_ocr
    logger.info(
        f"run_step30: ocr.transcribe() produced {len(chunks)} chunks in {ocr_elapsed:.1f}s "
        f"({ocr_elapsed / max(len(pages), 1):.2f}s/page)"
    )

    if args.smoke:
        full_book_pages = len(loader.load_pages(cfg))
        projected_h = (ocr_elapsed / max(len(pages), 1)) * full_book_pages / 3600
        print("\n=== SMOKE TIMING PROJECTION ===")
        print(f"  measured: {ocr_elapsed / max(len(pages), 1):.2f}s/page over {len(pages)} pages")
        print(f"  full book: {full_book_pages} pages")

        # A smoke page that was already transcribed is a cache hit, not a measurement: it
        # costs ~0s and drags the projection toward zero. Step 30's first push learned this
        # the expensive way -- it resumed from a checkpoint, all 5 smoke pages were cached,
        # and the gate printed "0.00s/page -> ~0.00h -> fits with margin" while the real
        # cost was 15.26s/page and 4.4h. plan.md's "measure before committing" gate is
        # worthless if it can silently report the timing of work it never did, so refuse to
        # project instead of printing a number that looks like evidence.
        n_cached = sum(1 for p in pages if (OCR_DIR / f"{p.id}.mmd").exists())
        if n_cached:
            print(
                f"  ⚠ {n_cached}/{len(pages)} smoke pages were ALREADY TRANSCRIBED (cache "
                f"hits, ~0s each) -- this projection is NOT a valid measurement."
            )
            if n_cached == len(pages):
                print(
                    "  -> every smoke page was cached; the number above measures nothing. "
                    "To get a real s/page, point --limit-pages at pages with no .mmd yet, "
                    "or clear the cache for a few pages first."
                )
                return
            print("  -> treat the projection below as a LOWER BOUND only.")

        print(f"  PROJECTED full-book OCR time: ~{projected_h:.2f}h")
        print("  Kaggle ceiling (plan.md 11.4): ~9h interactive / ~12h commit")
        if projected_h > 9:
            print(
                "  -> projected time exceeds the interactive ceiling; run via 'Save & Run "
                "All (Commit)' and/or plan a multi-push split."
            )
        else:
            print("  -> projected time fits inside a single interactive session, with margin.")
        return

    hooks.run(hooks.AFTER_OCR, {"chunks": chunks})
    chunks = chunk.split(chunks, cfg)
    hooks.run(hooks.BEFORE_INDEX, {"chunks": chunks})
    vectors = embed.encode(chunks, cfg)
    store.build(chunks, vectors, cfg)

    logger.info(f"run_step30: full pipeline done in {time.time() - t0:.1f}s total")

    print("\n=== Step 30 validate() ===")
    all_pages = loader.load_pages(cfg)  # the REAL full set, not the (possibly limited) run above
    validate(all_pages)
    print("validate() passed -- word floor + split leakage + schema all clean.")

    meta_path = Path("data/index/index_meta.json")
    if meta_path.exists():
        print("\n=== index_meta.json (Section 5's numbers) ===")
        print(meta_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
