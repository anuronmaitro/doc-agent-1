# Data contract
- Put scanned page-images in `data/raw/` (gitignored).
- Minimum: **>=300 pages AND >=60,000 words** extracted (huge-corpus profile: >=100k pages).
- Record source URL + license in `configs/task.yaml`. Public-domain / openly licensed only.
- Splits by **document**, not page (leakage rule).

## OCR annotation conventions (Steps 19-24)

Three people are hand-correcting 164 pages (39 test + 20 val + 105 train). Read this
*before* annotating anything — inconsistent labels hurt more than fewer labels, and this
doc is the one place all three of us agree on the rules.

**LaTeX form**
- Prefer `\tfrac` for inline fractions, `\frac` for display/standalone equations —
  matches the reference gold pages already in `grading_kit/labels.jsonl` (243, 255, 360).
- Use plain `(` `)` for ordinary parens; reserve `\left(` `\right)` only where the
  original visibly uses a taller bracket (nested fractions, tall radicals).
- Keep spacing exactly as printed (A&S's own `\,` digit-grouping in constants like
  `1.77245\,38509` is part of the source — don't collapse it).

**What's part of the label**
- **The formula's equation number (e.g. `6.1.8`) is NOT part of `text`** — it goes in the
  chunk id (`vision/ocr.py` / `index/chunk.py` already parse it via `FORMULA_ID_RE`), not
  in the transcription. Don't retype it into the body.
- Structural headers (section titles, table captions) ARE part of `text`.

**Unreadable regions**
- Mark a genuinely illegible glyph/region inline as `[UNREADABLE]` rather than guessing —
  a wrong guess is worse than an honest gap; `eval.metrics` treats it as a normal token
  when scoring, so it costs char-F1 fairly rather than silently inflating the score.

**Running heads / footers**
- Drop them from `text` (page headers like "EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS"
  repeated at the top of every chapter page are not content); the printed folio itself is
  already carried as page metadata (`page_id`, `printed_page`), not re-typed into the body.

**Dense numeric tables**
- Do not transcribe every cell. Transcribe: the table's structural headers/column labels,
  the defining formula(s) at the foot of the table, and a **sampled subset of rows**
  (first/last few + a couple from the middle) — same convention as our A1 gold page 243.
  Note in a trailing comment-style line that rows were sampled, not exhaustive.

**Pages with NO baseline draft — read this before you start a batch**
> ⚠️ Step 16's real full-book run measured a much higher failure rate than expected:
> **446/1040 pages (42.9%) produced no usable baseline transcript** — `data/ocr/failures.json`
> breaks this down as `empty-or-near-empty` (314), `nougat-missing-page-marker` (82), and
> `repetition-degeneration` (50). Each annotation batch in Steps 19-24 lists exactly which
> of its pages fall in this bucket.
>
> For those pages there is **nothing to correct** — the "correct, don't retype" workflow
> does not apply. Treat them as **blank-slate transcription** (read the scan, type the
> LaTeX from scratch) and budget time accordingly; do not skip them or leave `text` empty.
>
> Every per-page annotation JSON (`data/annot/val/*.json`, `data/annot/train/*.json`,
> written by `scripts/annotate_helper.py`) carries two fields sourced directly from
> `data/ocr/failures.json`, so this distinction survives into the fine-tune data and
> Step 29's before/after reporting instead of being silently merged with corrected pages:
> ```json
> {"baseline_failed": true, "baseline_reason": "empty-or-near-empty"}
> ```
> `baseline_failed: false` / `baseline_reason: null` means a real draft existed and was
> corrected rather than typed from scratch.
