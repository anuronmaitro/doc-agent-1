"""Step 29 — Final TEST measurement (the ONE-TIME opening -- run TWICE, deliberately).

Runs three readers over the 39 TEST pages in grading_kit/labels.jsonl:
  - "baseline": pretrained facebook/nougat-base
  - "finetuned": base + data/models/ocr_lora/curve_n122 LoRA adapter -- the ORIGINAL,
    one-time Step 29 measurement (plan.md's original "use curve_n122" guidance).
  - "hybrid": the REGION-ROUTING hybrid added at Step 28 point 11 (curve_n122 whole-page
    + table_ft applied only to detected table-region crops, via
    KAGGLE/region_routing_check/validate_region_routing.py's build_hybrid_prediction()) --
    a SECOND, deliberate TEST-set measurement, decided explicitly by the team
    (plan.md Step 29's 2026-08-13 update) to evaluate a reader that only had validation
    evidence before now. This is a KNOWN deviation from the "one-time opening" rule --
    report BOTH "finetuned" and "hybrid" in the final table, never overwrite one with
    the other (plan.md: "append, don't replace").

Scoring mirrors plan.md Step 29's own instructions exactly, for all three readers:
  - Failure rate is a first-class number, not a footnote (vision.ocr._failure_reason,
    the SAME detector — widened at Step 28 to catch block-level repetition spirals — as
    every prior step since 16, so this number is comparable to Step 16/18b/21/28's own).
  - char-F1 (eval.metrics.ocr_f1) and exact-match (eval.metrics.exact_formula_match) are
    only computed over pages that produced a transcript (a 0-length prediction isn't a
    meaningful char-F1 sample).
  - exact-match is rolled up FORMULA-WEIGHTED (weight = len(extract_formulas(gold))), so
    a 0-formula table/prose page contributes zero weight instead of a punishing zero
    score (plan.md's own note, credited to S3's Step 12 finding).

Two-phase design: `run()`/`run_hybrid()` call the models and write raw per-page results
to reports/step29_test_eval_results.json (including full pred_text, same discipline as
Step 28's run_state.json) — `report()` re-scores/summarizes from that JSON without
regenerating anything, so a fixed detector or a report-formatting change never requires
re-running the (slow, GPU, one-time-except-this-once) generation pass.

Usage (each reader is its own process invocation -- see run()'s docstring for why
baseline/finetuned never share a process; hybrid depends on "finetuned" already being in
the results file, since it reuses those predictions rather than regenerating them):
    python scripts/run_test_eval.py --run baseline
    python scripts/run_test_eval.py --run finetuned
    python scripts/run_test_eval.py --run hybrid          # needs "finetuned" already done
    python scripts/run_test_eval.py --run baseline --limit 3    # timing probe
    python scripts/run_test_eval.py --report                    # (re)build the report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "KAGGLE" / "region_routing_check"))

from doc_agent.eval.metrics import (  # noqa: E402
    exact_formula_match,
    extract_formulas,
    ocr_f1,
)
from doc_agent.vision.ocr import (  # noqa: E402
    MAX_NEW_TOKENS,
    REPETITION_PENALTY,
    Reader,
    _failure_reason,
)

LABELS_PATH = Path("grading_kit/labels.jsonl")
HELDOUT_DIR = Path("grading_kit/heldout_pages")
TRAIN_CFG_PATH = Path("configs/train_ocr.yaml")
ADAPTER_DIR = Path("data/models/ocr_lora/curve_n122")  # "finetuned" phase -- plan.md's
# ORIGINAL guidance, not table_ft
RESULTS_PATH = Path("reports/step29_test_eval_results.json")
REPORT_PATH = Path("reports/step29_test_eval.md")

EXPECTED_CHAR_F1 = (0.88, 0.93)  # summary.md §4e
EXPECTED_EXACT_MATCH = (0.55, 0.75)  # summary.md §4e


def load_test_pages() -> list[dict[str, Any]]:
    rows = []
    with open(LABELS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["image_path"] = str(HELDOUT_DIR / f"{row['page_id']}.png")
            if not Path(row["image_path"]).exists():
                raise FileNotFoundError(row["image_path"])
            rows.append(row)
    return rows


def _load_finetuned(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Base model + the accepted curve_n122 LoRA adapter, loaded for inference only."""
    from peft import PeftModel
    from transformers import NougatProcessor, VisionEncoderDecoderModel

    model_name = cfg["ocr"]["model"]
    revision = cfg["ocr"].get("revision")
    processor = NougatProcessor.from_pretrained(model_name, revision=revision)
    base = VisionEncoderDecoderModel.from_pretrained(model_name, revision=revision)
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR), is_trainable=False)
    model.eval()
    return processor, model


def _generate_finetuned(processor: Any, model: Any, image: Any) -> str:
    """Mirrors vision.ocr.Reader._generate()'s exact decoding params, so the fine-tuned
    number is comparable to the baseline's, not partly a decoding-config difference."""
    import torch

    pixel_values = processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        try:
            generate_fn = model.generate
        except AttributeError:  # pragma: no cover - PeftModel delegates in practice
            generate_fn = model.base_model.model.generate
        outputs = generate_fn(
            pixel_values,
            min_length=1,
            max_new_tokens=MAX_NEW_TOKENS,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            repetition_penalty=REPETITION_PENALTY,
        )
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return str(processor.post_process_generation(text, fix_markdown=False))


def _score(pred: str, gold: str) -> dict[str, Any]:
    """Block-level spiral check happens FIRST (via _failure_reason, which now includes
    Step 28's _has_duplicate_block) -- a page whose 'success' is duplicated garbage must
    not reach ocr_f1/exact_formula_match at all, on EITHER run, per plan.md's warning."""
    reason = _failure_reason(pred)
    if reason is not None:
        return {"failed": True, "failure_reason": reason}
    gold_formulas = extract_formulas(gold)
    return {
        "failed": False,
        "char_f1": ocr_f1(pred, gold),
        "exact_match": exact_formula_match(pred, gold),
        "formula_weight": len(gold_formulas),
    }


def run(phase: str, limit: int = 0) -> None:
    """Run ONE reader (`phase` = "baseline" or "finetuned") over the test pages.

    Deliberately split into two separate process invocations rather than one script
    holding both models in memory at once: loading two full ~350M-param
    VisionEncoderDecoderModel instances simultaneously (baseline + a second base model
    under the PEFT wrapper) exhausted this machine's Windows page file
    (`OSError: The paging file is too small for this operation to complete`, and in one
    run a hard segfault before that message could even be printed) -- measured, not
    assumed, by isolating baseline-only and fine-tuned-only loads (both fine
    individually) before finding the combined case failing. Each phase writes into its
    own key of the SAME results JSON, so `--phase baseline` then `--phase finetuned`
    (in two separate `python` calls) accumulates safely.
    """
    from PIL import Image as PILImage

    assert phase in ("baseline", "finetuned")
    pages = load_test_pages()
    if limit:
        pages = pages[:limit]

    cfg = yaml.safe_load(TRAIN_CFG_PATH.read_text(encoding="utf-8"))
    if phase == "baseline":
        print("loading baseline reader (facebook/nougat-base)...")
        reader = Reader({"ocr": cfg["ocr"], "device": "cpu"})
    else:
        print(f"loading fine-tuned reader ({ADAPTER_DIR})...")
        processor, ft_model = _load_finetuned(cfg)

    results: dict[str, Any] = {"baseline": [], "finetuned": []}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    done_ids = {r["page_id"] for r in results.get(phase, [])}
    if done_ids and not limit:
        pages = [p for p in pages if p["page_id"] not in done_ids]
        print(f"resuming {phase}: {len(done_ids)} pages already scored, {len(pages)} remaining")

    for rec in pages:
        image = PILImage.open(rec["image_path"]).convert("RGB")
        gold = rec["text"]

        t0 = time.time()
        if phase == "baseline":
            pred, _conf = reader._generate(image)
        else:
            pred = _generate_finetuned(processor, ft_model, image)
        elapsed = time.time() - t0
        row = {
            "page_id": rec["page_id"],
            "region_type": rec["region_type"],
            "pred_text": pred,
            "gen_seconds": elapsed,
            **_score(pred, gold),
        }

        results[phase] = [r for r in results[phase] if r["page_id"] != rec["page_id"]] + [row]
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

        status = (
            f"FAIL:{row.get('failure_reason')}" if row["failed"] else f"f1={row['char_f1']:.3f}"
        )
        print(f"{rec['page_id']} ({rec['region_type']}): {phase} {status} ({elapsed:.1f}s)")

    print(f"\nwrote {RESULTS_PATH} ({phase}: {len(results[phase])} pages)")


def run_hybrid(limit: int = 0) -> None:
    """Region-routing hybrid pass -- the SECOND, deliberate TEST-set measurement
    (plan.md Step 29's 2026-08-13 update). Reuses "finetuned" (curve_n122)'s own
    predictions from RESULTS_PATH rather than regenerating them (build_hybrid_prediction
    only needs curve_n122's text + whether it failed, not a fresh model call), and only
    calls table_ft at all for (a) the whole-page fallback on pages where curve_n122
    already failed, and (b) crop-level generation on detected table regions -- see
    KAGGLE/region_routing_check/validate_region_routing.py's module docstring for the
    two real splice bugs this already had fixed, both inherited for free by reusing that
    module's build_hybrid_prediction() rather than reimplementing the splice logic here.
    """
    import torch
    from PIL import Image as PILImage
    from validate_region_routing import (  # type: ignore[import-not-found]
        LAYOUT_CFG,
        build_hybrid_prediction,
        build_lit_with_table_ft,
    )

    from doc_agent.contracts import Page  # noqa: E402
    from doc_agent.vision import layout  # noqa: E402

    pages = load_test_pages()
    if limit:
        pages = pages[:limit]

    results: dict[str, Any] = {"baseline": [], "finetuned": [], "hybrid": []}
    if RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    curve_by_id = {r["page_id"]: r for r in results.get("finetuned", [])}
    missing = [p["page_id"] for p in pages if p["page_id"] not in curve_by_id]
    if missing:
        raise SystemExit(
            f"run_hybrid: {len(missing)} pages have no 'finetuned' (curve_n122) result "
            f"yet -- run `--run finetuned` first (it's the base this hybrid splices "
            f"onto): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    results.setdefault("hybrid", [])
    done_ids = {r["page_id"] for r in results["hybrid"]}
    if done_ids and not limit:
        pages = [p for p in pages if p["page_id"] not in done_ids]
        print(f"resuming hybrid: {len(done_ids)} pages already scored, {len(pages)} remaining")
    if not pages:
        print("hybrid: nothing left to do")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    cfg = yaml.safe_load(TRAIN_CFG_PATH.read_text(encoding="utf-8"))
    cfg["device"] = device
    max_new_tokens = cfg["data"]["max_target_length"]

    print("running layout.detect() (TATR enabled) over the pages needing a hybrid pass...")
    layout_pages = [Page(id=p["page_id"], image_path=p["image_path"], doc_id="test") for p in pages]
    regions_by_page: dict[str, list[Any]] = {}
    for r in layout.detect(layout_pages, LAYOUT_CFG):
        regions_by_page.setdefault(r.page_id, []).append(r)

    print("loading table_ft adapter for crop/fallback generation...")
    table_ft_lit = build_lit_with_table_ft(cfg, device)

    for rec in pages:
        pid = rec["page_id"]
        gold = rec["text"]
        curve_row = curve_by_id[pid]
        curve_failed = bool(curve_row["failed"])

        table_ft_whole_page = None
        if curve_failed:
            # Fix 1 (validate_region_routing.py): only needed as a fallback when
            # curve_n122's own base already failed -- lazy, not generated for every page.
            from run_finetune import _generate_page  # noqa: E402

            image = PILImage.open(rec["image_path"]).convert("RGB")
            table_ft_whole_page = _generate_page(table_ft_lit, image, device, max_new_tokens)

        t0 = time.time()
        pred = build_hybrid_prediction(
            page_id=pid,
            image_path=rec["image_path"],
            page_regions=regions_by_page.get(pid, []),
            curve_pred_text=curve_row.get("pred_text", ""),
            curve_failed=curve_failed,
            table_ft_whole_page_pred_text=table_ft_whole_page,
            table_ft_lit=table_ft_lit,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.time() - t0
        row = {
            "page_id": pid,
            "region_type": rec["region_type"],
            "pred_text": pred,
            "gen_seconds": elapsed,
            **_score(pred, gold),
        }

        results["hybrid"] = [r for r in results["hybrid"] if r["page_id"] != pid] + [row]
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

        status = (
            f"FAIL:{row.get('failure_reason')}" if row["failed"] else f"f1={row['char_f1']:.3f}"
        )
        print(f"{pid} ({rec['region_type']}): hybrid {status} ({elapsed:.1f}s)")

    print(f"\nwrote {RESULTS_PATH} (hybrid: {len(results['hybrid'])} pages)")


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    failed = [r for r in rows if r["failed"]]
    ok = [r for r in rows if not r["failed"]]
    exact_num = sum(r["exact_match"] * r["formula_weight"] for r in ok)
    exact_den = sum(r["formula_weight"] for r in ok)
    return {
        "n_pages": n,
        "n_failed": len(failed),
        "failure_rate": len(failed) / n if n else 0.0,
        "mean_char_f1_among_successes": (sum(r["char_f1"] for r in ok) / len(ok) if ok else 0.0),
        "formula_weighted_exact_match": (exact_num / exact_den) if exact_den else 0.0,
    }


def _summarize_by_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    types = sorted({r["region_type"] for r in rows})
    return {t: _summarize([r for r in rows if r["region_type"] == t]) for t in types}


def _worst_failure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The single worst-scoring page: an outright failure ranks worse than any low
    char-F1 (plan.md: 'produced nothing' is a legitimate, complete worst-failure answer),
    so failures sort first, then by ascending char-F1 among successes."""
    failed = [r for r in rows if r["failed"]]
    if failed:
        return failed[0]
    return min(rows, key=lambda r: r["char_f1"])


def report() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    base_rows, ft_rows = data["baseline"], data["finetuned"]
    hybrid_rows = data.get("hybrid", [])
    assert len(base_rows) == len(ft_rows) == 39, (
        f"expected all 39 test pages scored, got {len(base_rows)}/{len(ft_rows)} -- "
        "re-run `--run` (it resumes) before building the report"
    )
    has_hybrid = len(hybrid_rows) == 39

    base_summary = _summarize(base_rows)
    ft_summary = _summarize(ft_rows)
    base_by_type = _summarize_by_type(base_rows)
    ft_by_type = _summarize_by_type(ft_rows)
    worst = _worst_failure(ft_rows)
    if has_hybrid:
        hybrid_summary = _summarize(hybrid_rows)
        hybrid_by_type = _summarize_by_type(hybrid_rows)
        worst_hybrid = _worst_failure(hybrid_rows)

    lines = ["# Step 29 — final TEST measurement (baseline vs. fine-tuned)", ""]
    lines.append(
        "The ONE-TIME opening of the 39 TEST pages — every design decision above this "
        "was already made on the 20 validation pages (Steps 21/28)."
    )
    if has_hybrid:
        lines.append("")
        lines.append(
            "> ⚠️ **This report includes a SECOND, deliberate TEST-set measurement** "
            "(the region-routing hybrid, plan.md Step 29's 2026-08-13 update), decided "
            "explicitly by the team to evaluate a reader that only had validation-set "
            "evidence before now (Step 28 point 11). Both the original one-time "
            "`curve_n122`-only result and this second, compromised-by-design "
            "measurement are reported below, side by side — the second is APPENDED, "
            "not a replacement for the first."
        )
    lines.append("")
    lines.append("## Before / after — 39 test pages")
    lines.append("")
    header = "| metric | baseline (pretrained) | fine-tuned (curve_n122) |"
    sep = "|---|---|---|"
    if has_hybrid:
        header += " fine-tuned (region-routing hybrid — 2nd TEST look) |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)
    row = (
        f"| failure rate | {base_summary['n_failed']}/39 "
        f"({base_summary['failure_rate']:.1%}) | {ft_summary['n_failed']}/39 "
        f"({ft_summary['failure_rate']:.1%}) |"
    )
    if has_hybrid:
        row += f" {hybrid_summary['n_failed']}/39 ({hybrid_summary['failure_rate']:.1%}) |"
    lines.append(row)
    row = (
        f"| char-F1 (successes) | {base_summary['mean_char_f1_among_successes']:.3f} | "
        f"{ft_summary['mean_char_f1_among_successes']:.3f} |"
    )
    if has_hybrid:
        row += f" {hybrid_summary['mean_char_f1_among_successes']:.3f} |"
    lines.append(row)
    row = (
        f"| exact formula match (formula-weighted) | "
        f"{base_summary['formula_weighted_exact_match']:.3f} | "
        f"{ft_summary['formula_weighted_exact_match']:.3f} |"
    )
    if has_hybrid:
        row += f" {hybrid_summary['formula_weighted_exact_match']:.3f} |"
    lines.append(row)
    lines.append("")

    lines.append("## Per-page-type breakdown")
    lines.append("")
    lines.append(
        "| region_type | n | baseline failure rate | baseline char-F1 | baseline exact | "
        "fine-tuned failure rate | fine-tuned char-F1 | fine-tuned exact |"
        + (" hybrid failure rate | hybrid char-F1 | hybrid exact |" if has_hybrid else "")
    )
    lines.append("|---|---|---|---|---|---|---|---|" + ("---|---|---|" if has_hybrid else ""))
    for t in sorted(base_by_type):
        b, f = base_by_type[t], ft_by_type[t]
        row = (
            f"| {t} | {b['n_pages']} | {b['failure_rate']:.1%} | "
            f"{b['mean_char_f1_among_successes']:.3f} | "
            f"{b['formula_weighted_exact_match']:.3f} | {f['failure_rate']:.1%} | "
            f"{f['mean_char_f1_among_successes']:.3f} | "
            f"{f['formula_weighted_exact_match']:.3f} |"
        )
        if has_hybrid:
            h = hybrid_by_type[t]
            row += (
                f" {h['failure_rate']:.1%} | {h['mean_char_f1_among_successes']:.3f} | "
                f"{h['formula_weighted_exact_match']:.3f} |"
            )
        lines.append(row)
    lines.append("")

    lines.append("## Worst failure")
    lines.append("")
    lines.append("**Fine-tuned (curve_n122, original one-time measurement):**")
    if worst["failed"]:
        lines.append(
            f"**{worst['page_id']}** ({worst['region_type']}) — outright failure, "
            f"reason: `{worst['failure_reason']}`. Produced no usable transcript at all."
        )
    else:
        lines.append(
            f"**{worst['page_id']}** ({worst['region_type']}) — lowest char-F1 among "
            f"pages that produced output: {worst['char_f1']:.3f}."
        )
    if has_hybrid:
        lines.append("")
        lines.append("**Fine-tuned (region-routing hybrid, 2nd TEST look):**")
        if worst_hybrid["failed"]:
            lines.append(
                f"**{worst_hybrid['page_id']}** ({worst_hybrid['region_type']}) — "
                f"outright failure, reason: `{worst_hybrid['failure_reason']}`. "
                "Produced no usable transcript at all."
            )
        else:
            lines.append(
                f"**{worst_hybrid['page_id']}** ({worst_hybrid['region_type']}) — "
                f"lowest char-F1 among pages that produced output: "
                f"{worst_hybrid['char_f1']:.3f}."
            )
    lines.append("")

    lines.append("## Sanity check vs. summary.md §4e")
    lines.append("")
    sanity_targets = [("fine-tuned (curve_n122)", ft_summary)]
    if has_hybrid:
        sanity_targets.append(("fine-tuned (region-routing hybrid)", hybrid_summary))
    for label, summary in sanity_targets:
        f1 = summary["mean_char_f1_among_successes"]
        ex = summary["formula_weighted_exact_match"]
        f1_in_band = EXPECTED_CHAR_F1[0] <= f1 <= EXPECTED_CHAR_F1[1]
        ex_in_band = EXPECTED_EXACT_MATCH[0] <= ex <= EXPECTED_EXACT_MATCH[1]
        lines.append(f"**{label}:**")
        lines.append(
            f"- Expected char-F1 {EXPECTED_CHAR_F1[0]}-{EXPECTED_CHAR_F1[1]}; measured "
            f"{f1:.3f} — {'IN BAND' if f1_in_band else 'OUT OF BAND, investigate'}."
        )
        lines.append(
            f"- Expected exact-match {EXPECTED_EXACT_MATCH[0]}-{EXPECTED_EXACT_MATCH[1]}; "
            f"measured {ex:.3f} — {'IN BAND' if ex_in_band else 'OUT OF BAND, investigate'}."
        )
        lines.append(
            "- Those bands describe pages that produce *some* output; they say nothing "
            f"about failure rate ({summary['failure_rate']:.1%} here), so landing inside "
            "the char-F1 band while still failing outright on some pages is plausible, "
            "not a contradiction (plan.md Step 29 point 5)."
        )
        lines.append("")

    if not has_hybrid:
        lines.append(
            "_(Region-routing hybrid not yet scored — run `--run hybrid` after "
            "`--run finetuned`, then `--report` again to add the second TEST-set look.)_"
        )
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        choices=["baseline", "finetuned", "hybrid"],
        help="generate one reader's predictions (each is its own process invocation -- "
        "see run()/run_hybrid()'s docstrings for why; hybrid needs 'finetuned' already "
        "scored in the results file)",
    )
    p.add_argument("--report", action="store_true", help="(re)build the report from results")
    p.add_argument("--limit", type=int, default=0, help="0 = all 39 pages")
    args = p.parse_args()

    if not args.run and not args.report:
        p.error("pass --run {baseline,finetuned,hybrid} and/or --report")
    if args.run == "hybrid":
        run_hybrid(limit=args.limit)
    elif args.run:
        run(args.run, limit=args.limit)
    if args.report:
        report()


if __name__ == "__main__":
    main()
