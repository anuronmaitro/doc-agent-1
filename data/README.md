# Data contract
- Put scanned page-images in `data/raw/` (gitignored).
- Minimum: **>=300 pages AND >=60,000 words** extracted (huge-corpus profile: >=100k pages).
- Record source URL + license in `configs/task.yaml`. Public-domain / openly licensed only.
- Splits by **document**, not page (leakage rule).

## OCR annotation conventions (Steps 19-24)

Three people are hand-correcting 181 pages (39 test + 20 val + 122 train -- train grew
105->122 at the Step 18b gate: the repaired reader's full-book run still missed 28.5% of
pages, above the 25% gate line, so 17 pages were added from the chapters it failed hardest
on). Read this
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
- **Correction (2026-08-10, Step 19): the formula's equation number (e.g. `6.1.8`) IS part
  of `text`, contradicting an earlier draft of this rule.** `eval.metrics.extract_formulas()`
  parses the id straight OUT of `text` with `FORMULA_ID_RE` — the id has to be there for
  extraction to find it at all, that's not a separate side-channel. The 3 original gold
  pages (243, 255, 360) already did this (e.g. `"6.1.1  \Gamma(z)=..."`), and Step 19's 18
  pages follow the same precedent for consistency within `labels.jsonl`. Keep the id inline,
  followed by two spaces, then the formula.
- Structural headers (section titles, table captions) ARE part of `text`.

**Unreadable regions**
- Mark a genuinely illegible glyph/region inline as `[UNREADABLE]` rather than guessing —
  a wrong guess is worse than an honest gap; `eval.metrics` treats it as a normal token
  when scoring, so it costs char-F1 fairly rather than silently inflating the score.

**Running heads / footers**
- **Correction (2026-08-10, Step 19): KEEP the chapter running head + page number as the
  first line of `text`**, contradicting an earlier draft of this rule — e.g.
  `"EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.229)"`. All 3 original gold pages do this
  (`"EXPONENTIAL INTEGRAL AND RELATED FUNCTIONS (p.243)"`, `"BESSEL FUNCTIONS OF INTEGER
  ORDER (p.360)"`), so Step 19 matched that precedent rather than the written rule below it,
  for the same reason as the equation-number correction above: consistency within
  `labels.jsonl` matters more than a rule the first 3 pages never actually followed. Genuine
  page footers with no chapter/page information (rare in A&S) can still be dropped.

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

## Step 25 — NIST Stage A pairs (`data/annot/nist/`)

`scripts/extract_nist_pairs.py` (config: `configs/nist_extract.yaml`) renders `(image,
text)` pairs from the NIST Handbook of Mathematical Functions (2010) PDF — a companion
volume to the A&S 1964 handbook, LaTeX-typeset via pdfTeX, public domain. Output:
`data/annot/nist/images/*.png` (one crop per formula) + `data/annot/nist/pairs.jsonl`
(one record per crop: `pair_id`, `pdf_page`, `eqn_id`, `text`, `image`).

**Measured result: 695 pairs (44 containing `\frac`), 4.9 MB total — committed outright**
(same call as `data/ocr/`; see `.gitignore`'s comment). Below the 5,000-6,000 target in
plan.md Step 25, and this is a deliberate, re-verified outcome, not a shortfall to
silently chase by loosening rules — read the whole section below before touching the
thresholds in `configs/nist_extract.yaml`.

**Why this deviates from plan.md's premise, and what "exact by construction" actually
means here.** Step 25 describes the source as "LaTeX-native, so labels are exact by
construction." pdfTeX does not embed LaTeX source in the PDF, only absolute-positioned
glyphs grouped into per-baseline lines and, within those, size/font-tagged spans — there
is no `\frac` or `^` anywhere in the content stream. Two structures ARE reconstructed from
that raw geometry, each confirmed against real rendered crops (not just plausible-looking
text) before being trusted:

- **Superscript/subscript** — a span whose font size sits well below its line's dominant
  size, offset above or below the dominant baseline, is a superscript/subscript (e.g. the
  "s" in `f_s(z)` is CMMI7 6.97pt, ~1.5pt below the CMMI10 9.96pt baseline around it).
  Without this, `x^n` flattened to the plain, structurally-wrong string `"xn"`.
- **Fractions** — pdfTeX draws a fraction's vinculum as a real thin vector stroke,
  independent of the glyphs above/below it. `page.get_drawings()` finds that stroke
  directly, so `\frac{num}{den}` is only emitted when a matching bar is physically present
  at the right position — never inferred from glyph layout alone. A formula continuing
  past the fraction on the denominator's own baseline (`"dw/dz + a^2w^2 = 1,"`) is split by
  comparing each glyph's x-position to the bar's x-range.

Still **not** reconstructed, on purpose: stacked sums/integrals/products (drawn with the
CMEX font — see below — with limits at arbitrary offsets and no vinculum-equivalent to
confirm the pairing) and radicals (a `√`'s extent isn't delimited by anything in the
content stream). Guessing either would reproduce the exact silent-wrong-label failure the
whole rest of this section is about avoiding — left as a documented gap.

**Failure modes found (and fixed) while building this — worth knowing before touching the
script, since several look "done" until checked against the actual crop:**

1. **A tall radical decoded as the literal letter "p".** pdfTeX's CMEX10 font (stretchy
   delimiters, big radicals/operators, more below) has an unreliable Unicode ToUnicode
   CMap. Any fragment touching a `CMEX*` font is excluded outright.
2. **A fraction (`dw/dz`) flattened to `"dw dz"`** before fraction reconstruction existed —
   numerator/denominator can sit close enough in y to slip past row-clustering; caught by
   requiring same-row fragments to never overlap in x.
3. **A wrapped prose citation `"(4.45.10)"` collided with the real equation 4.45.10's
   id-line**, silently overwriting its crop on disk (same `pair_id`, last write wins).
   Fixed by requiring a genuine label to be bare (no parens), plus a hard collision guard
   regardless of cause.
4. **An overline accent (`z̄`, complex conjugate) mistaken for a fraction vinculum** — both
   are zero-height vector strokes; the accent is only ~5pt wide vs. a real vinculum's
   ~12.6pt+ even over a single-character numerator. Merged two DIFFERENT equations
   (1.9.12 and 1.9.13) into one bogus fraction until `_fraction_bars` required width >= 8pt.
5. **An isolated fragment near an id treated as if it were the whole formula** — a big
   fraction's numerator/denominator can sit outside `walk_gap_pt` in every direction,
   leaving only a stray nearby piece (a lone `"0"` off an integral sign; a fraction's bare
   denominator with no numerator or "lhs =" prefix). Fixed with a plausibility gate: every
   real numbered display in this book states a relation, so the final `text` must contain
   an `=`/`<`/`≤`/etc. symbol, and must not itself start with `=` (a sign the id labels
   only the tail of a larger multi-line expression, as on p.776 eqn 34.3.14).
6. **A numerator's own line carrying extra prefix content** (`t^{µ-1} * t^{-s-α} =` sharing
   the numerator's baseline) spliced into the wrong place. Fixed by requiring the
   numerator's line contain ONLY bar-width content — a prefix/tail is only supported on
   the denominator's side.
7. **A superscript split from its own base across the bar-x-range boundary**, leaving an
   orphaned `^{a}` with nothing before it. Fixed by rejecting any reconstruction whose
   prefix/tail/numerator/denominator starts with a bare `^{`/`_{` marker.
8. **Two unrelated same-row fragments 111pt apart in x, joined with a single space** —
   `"| tanh z| ="` and a stray `"."` from a much bigger fraction display whose actual
   numerator/denominator fell outside `walk_gap_pt`, glued into the wrong, truncated
   `"| tanh z| = ."`. Fixed by requiring same-row fragments to be within `max_frag_gap_pt`
   of each other before joining, not just present in the same row.

Every one of the above was caught by rendering the actual crop and reading it, not by
trusting the text output looked plausible — several (5, 6, 8) produced text that read as a
complete, well-formed formula and were wrong anyway. **If you change the thresholds in
`configs/nist_extract.yaml`, re-render a random sample of crops and look at them before
trusting the new count.**

The extraction is still deliberately conservative: multi-row constructs beyond a single
confirmed fraction (stacked sums, matrices, anything genuinely multi-line) are detected
and **skipped**, not scrambled. In a 967-page special-functions reference most display
equations are multi-row, so 695 is the realistic ceiling for this safe-by-construction
approach at its current sophistication, not a bug to chase down by loosening acceptance
rules.

Two known, smaller, left-as-is limitations, both confirmed rare enough not to be worth the
same reconstruction effort as fractions/scripts:

- An overline accent's own bar is invisible to text extraction (it's the vector stroke
  described in failure 4 above, not a character), so a genuinely bar-accented variable
  like `z̄` extracts as plain `z` with the diacritic silently dropped.
- A small number of pages instead extract the accent as a literal Unicode macron
  character sitting before the base letter (`"me_ν(z, q) = me_¯ν(−¯z, ¯q)"`), which is
  honest but not idiomatic LaTeX (`\bar{ν}`).

**Format, and what Step 26/27 need to know about it.** `text` keeps the id inline first
(`"5.5.3  Γ(z) Γ(1 − z) = π/ sin(πz), ..."`) — same inline-id convention as
`data/annot/train|val`, so `eval.metrics.extract_formulas()`'s `FORMULA_ID_RE` parses it
the same way. Reconstructed superscripts/subscripts/fractions use real LaTeX markup
(`^{...}`, `_{...}`, `\frac{...}{...}`), matching the hand-annotated `data/annot/train|val`
convention for those constructs specifically — but coverage is narrower than hand
annotation: no `\Gamma`, `\sum`, `\int`, `\sqrt`, or other named macros (those symbols
still extract as their literal Unicode character, e.g. `Γ` not `\Gamma`, since pdfTeX's own
Unicode CMap already resolves them correctly and wrapping a correctly-resolved character
in an unnecessary macro would be pure invention). Whoever builds the Stage A + Stage B
mixture in Step 27 should know Stage A is real, trainable LaTeX-flavored markup for the
constructs it covers, not a lesser dialect, but it is narrower in scope than Stage B's
full hand-corrected convention.
