# Step 29 — final TEST measurement (baseline vs. fine-tuned)

The ONE-TIME opening of the 39 TEST pages — every design decision above this was already made on the 20 validation pages (Steps 21/28).

> ⚠️ **This report includes a SECOND, deliberate TEST-set measurement** (the region-routing hybrid, plan.md Step 29's 2026-08-13 update), decided explicitly by the team to evaluate a reader that only had validation-set evidence before now (Step 28 point 11). Both the original one-time `curve_n122`-only result and this second, compromised-by-design measurement are reported below, side by side — the second is APPENDED, not a replacement for the first.

## Before / after — 39 test pages

| metric | baseline (pretrained) | fine-tuned (curve_n122) | fine-tuned (region-routing hybrid — 2nd TEST look) |
|---|---|---|---|
| failure rate | 20/39 (51.3%) | 9/39 (23.1%) | 3/39 (7.7%) |
| char-F1 (successes) | 0.441 | 0.423 | 0.443 |
| exact formula match (formula-weighted) | 0.000 | 0.016 | 0.014 |

## Per-page-type breakdown

| region_type | n | baseline failure rate | baseline char-F1 | baseline exact | fine-tuned failure rate | fine-tuned char-F1 | fine-tuned exact | hybrid failure rate | hybrid char-F1 | hybrid exact |
|---|---|---|---|---|---|---|---|---|---|---|
| formula | 18 | 33.3% | 0.416 | 0.000 | 11.1% | 0.477 | 0.020 | 0.0% | 0.500 | 0.018 |
| prose+formula | 7 | 42.9% | 0.504 | 0.000 | 14.3% | 0.342 | 0.000 | 0.0% | 0.385 | 0.000 |
| table | 14 | 78.6% | 0.458 | 0.000 | 42.9% | 0.377 | 0.000 | 21.4% | 0.387 | 0.000 |

## Worst failure

**Fine-tuned (curve_n122, original one-time measurement):**
**as_p0243** (table) — outright failure, reason: `nougat-missing-page-marker`. Produced no usable transcript at all.

**Fine-tuned (region-routing hybrid, 2nd TEST look):**
**as_p0243** (table) — outright failure, reason: `empty-or-near-empty`. Produced no usable transcript at all.

## Sanity check vs. summary.md §4e

**fine-tuned (curve_n122):**
- Expected char-F1 0.88-0.93; measured 0.423 — OUT OF BAND, investigate.
- Expected exact-match 0.55-0.75; measured 0.016 — OUT OF BAND, investigate.
- Those bands describe pages that produce *some* output; they say nothing about failure rate (23.1% here), so landing inside the char-F1 band while still failing outright on some pages is plausible, not a contradiction (plan.md Step 29 point 5).

**fine-tuned (region-routing hybrid):**
- Expected char-F1 0.88-0.93; measured 0.443 — OUT OF BAND, investigate.
- Expected exact-match 0.55-0.75; measured 0.014 — OUT OF BAND, investigate.
- Those bands describe pages that produce *some* output; they say nothing about failure rate (7.7% here), so landing inside the char-F1 band while still failing outright on some pages is plausible, not a contradiction (plan.md Step 29 point 5).
