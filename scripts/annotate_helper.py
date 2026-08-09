"""Step 17 — annotation helper (correct-don't-retype), used by Steps 19-24.

Given a list of PRINTED page numbers, writes one draft JSON per page into an output
folder (data/annot/val/ or data/annot/train/ — both committed, see .gitignore) so an
annotator corrects the model's own draft instead of retyping a page from scratch.

A page that Step 16's real baseline run logged in data/ocr/failures.json (empty output,
Nougat's own [MISSING_PAGE] marker, or a repetition-degeneration loop -- see
data/README.md "OCR annotation conventions") has NO usable draft. For those pages this
script writes baseline_failed=true + the failure reason instead of blank/garbage text, so
the annotator sees why there's nothing to correct rather than guessing -- and knows to
transcribe that page from scratch, not "fix" an empty string.

Usage (script):
    python scripts/annotate_helper.py --out data/annot/val --pages 332 334 340 ...
    python scripts/annotate_helper.py --out data/annot/train --pages 1 2 3 --preview 1

Usage (notebook cell / another script):
    from scripts.annotate_helper import bootstrap_annotations, render_side_by_side
    bootstrap_annotations([332, 334, 340], "data/annot/val")
    render_side_by_side("as_p0332")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from doc_agent.ingest.loader import _chapter_of  # noqa: E402
from doc_agent.vision.ocr import FAILURES_PATH, OCR_DIR, _page_image_path  # noqa: E402

# Ephemeral preview renders (side-by-side page + draft) live under data/interim/, which
# is already blanket-gitignored -- these are throwaway working aids for the annotator,
# not a deliverable, so they must never land in the committed data/annot/val|train/.
PREVIEW_DIR = Path("data/interim/annotate_preview")


def _page_id(printed_page: int) -> str:
    return f"as_p{printed_page:04d}"


def _load_failures() -> dict[str, dict]:
    if not FAILURES_PATH.exists():
        return {}
    try:
        return {
            row["page_id"]: row for row in json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
        }
    except (OSError, json.JSONDecodeError):
        return {}


def _draft_for(page_id: str, failures: dict[str, dict]) -> tuple[str, bool, str | None]:
    """Returns (draft_text, baseline_failed, baseline_reason)."""
    failure = failures.get(page_id)
    if failure is not None:
        return "", True, str(failure.get("reason", "unknown"))
    mmd_path = OCR_DIR / f"{page_id}.mmd"
    if mmd_path.exists():
        return mmd_path.read_text(encoding="utf-8"), False, None
    # Page was simply never run yet (not in failures.json, no cached .mmd) -- treat the
    # same as a failure so the annotator still gets an honest "nothing to correct" signal
    # instead of an empty draft that looks like a blank correction was intended.
    return "", True, "not-yet-transcribed"


def bootstrap_annotations(printed_pages: list[int], out_dir: str | Path) -> list[Path]:
    """Write one draft JSON per page into out_dir. Never overwrites an existing file --
    an annotator's in-progress or finished correction is never clobbered by a re-run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    failures = _load_failures()
    written: list[Path] = []

    for printed in printed_pages:
        page_id = _page_id(printed)
        dst = out / f"{page_id}.json"
        if dst.exists():
            written.append(dst)
            continue
        draft_text, baseline_failed, baseline_reason = _draft_for(page_id, failures)
        record = {
            "page_id": page_id,
            "printed_page": printed,
            "chapter_id": _chapter_of(printed),
            "baseline_failed": baseline_failed,
            "baseline_reason": baseline_reason,
            "draft_text": draft_text,
            "text": "",  # the annotator fills this in (corrected, or blank-slate transcribed)
            "annotator": None,
            "annotated_at": None,
        }
        dst.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(dst)

    return written


def render_side_by_side(page_id: str, *, save_to: str | Path | None = None) -> Path:
    """Save a page-image-beside-draft-text PNG so an annotator can compare at a glance.
    Shows the failure reason instead of a blank panel when the page has no draft."""
    import matplotlib.pyplot as plt
    from PIL import Image

    failures = _load_failures()
    draft_text, baseline_failed, baseline_reason = _draft_for(page_id, failures)
    image = Image.open(_page_image_path(page_id)).convert("L")

    fig, (ax_img, ax_txt) = plt.subplots(1, 2, figsize=(11, 7))
    ax_img.imshow(image, cmap="gray")
    ax_img.set_title(page_id)
    ax_img.axis("off")

    ax_txt.axis("off")
    if baseline_failed:
        ax_txt.text(
            0.5,
            0.5,
            f"NO BASELINE DRAFT\nreason: {baseline_reason}\n\n"
            "Transcribe from scratch (blank-slate),\nnot a correction.",
            ha="center",
            va="center",
            wrap=True,
            fontsize=11,
            color="firebrick",
        )
    else:
        ax_txt.text(0.0, 1.0, draft_text[:2000], ha="left", va="top", wrap=True, fontsize=8)
    ax_txt.set_title("model draft" if not baseline_failed else "baseline failed")

    fig.tight_layout()
    dst = Path(save_to) if save_to else PREVIEW_DIR / f"{page_id}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=100)
    plt.close(fig)
    return dst


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="data/annot/val or data/annot/train")
    p.add_argument("--pages", required=True, nargs="+", type=int, help="printed page numbers")
    p.add_argument("--preview", type=int, default=0, help="also render this many side-by-side PNGs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    written = bootstrap_annotations(args.pages, args.out)
    print(f"annotate_helper: {len(written)} draft JSON files ready under {args.out}/")
    for page in args.pages[: args.preview]:
        path = render_side_by_side(_page_id(page))
        print(f"  preview: {path}")


if __name__ == "__main__":
    main()
