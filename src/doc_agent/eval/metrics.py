"""Stage 9 — metrics"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..contracts import *  # noqa

# --- LaTeX normalisation ------------------------------------------------------------------
# Applied to BOTH sides before any comparison (plan.md Step 12). Our gold pages are
# hand-written LaTeX (grading_kit/labels.jsonl) while the reader emits Nougat's
# markdown+LaTeX, so without this we would be scoring notation style, not reading accuracy:
# `\tfrac12` and `\frac{1}{2}` are the same formula, and a model must not be marked wrong
# for choosing the other spelling. Every rule below collapses a *typographic* difference
# and none collapses a *mathematical* one -- `x^2` and `x^3` stay different.

# Math-mode delimiters and environments Nougat wraps display equations in.
_MATH_DELIMS = re.compile(r"\\\[|\\\]|\\\(|\\\)")
_ENVIRONMENTS = re.compile(r"\\(?:begin|end)\{[a-zA-Z*]+\}")
# Non-semantic annotations.
_LABELS = re.compile(r"\\(?:label|tag|ref|eqref|nonumber)\s*\{[^}]*\}")
# Explicit spacing: purely typographic in LaTeX, and the reader has no reason to reproduce
# our thin-space conventions. `\,` in particular is how the gold groups digits (1.45459\,66142).
_SPACING_MACROS = re.compile(r"\\[,;:!]|\\quad\b|\\qquad\b|\\hspace\s*\{[^}]*\}|\\ |(?<!\\)~")
# `\left(` / `\right)` only tell TeX how tall to draw a delimiter; the delimiter itself stays.
_LEFT_RIGHT = re.compile(
    r"\\(?:left|right|middle|bigl?|Bigl?|biggl?|Biggl?|bigr|Bigr|biggr|Biggr)\b"
)
# Ellipsis spellings.
_ELLIPSIS = re.compile(r"\\(?:ldots|cdots|dotsc|dotsb|dots)\b")
# Text-ish wrappers that all mean "upright operator name".
_UPRIGHT = re.compile(r"\\(?:operatorname|text|textrm|mathord)\s*\{")
# Single-token sub/superscript argument -> braced form, so `x^2` == `x^{2}` and
# `\int_0^\infty` == `\int_{0}^{\infty}`.
_SCRIPT_ARG = re.compile(r"([\^_])\s*(\\[a-zA-Z]+|[0-9A-Za-z])")
# `\frac12` -> `\frac{1}{2}` (reached after \tfrac/\dfrac are folded into \frac).
_FRAC_ARGS = re.compile(r"\\frac\s*(\\[a-zA-Z]+|[0-9A-Za-z])\s*(\\[a-zA-Z]+|[0-9A-Za-z])")
# Old-style TeX fractions, `{A \over B}`, are the same formula as `\frac{A}{B}` but share no
# characters with it -- Step 18b found Nougat's repaired (full-page) output uses `\over` 36
# times on a single gold page, so leaving it unfolded was silently costing real precision/
# recall credit for correct math. Innermost-brace-only per pass (no nested `{}` inside A or
# B), looped so a `{...{A \over B}...\over C}` outer fraction resolves after its inner one
# does -- one regex pass cannot see "the closing brace that matches THIS opening brace".
#
# Known limitation, accepted rather than chased: once an inner `\over` resolves to
# `\frac{A}{B}`, that substitution ITSELF introduces braces, so an outer `\over` sharing the
# same brace pair as a resolved inner one can no longer match (its content is no longer
# brace-free). Real-data check on the as_p0360 gold page: 36 raw `\over` occurrences, 25
# resolved this way, 11 left unresolved by this residue effect -- and precision/recall still
# improved measurably (0.417->0.458 / 0.752->0.833) because most `\over` usage in practice is
# NOT nested this deeply. Fully general resolution needs a recursive-descent brace parser,
# not a regex; not worth the complexity for the residual few percent this leaves on the table.
_OVER = re.compile(r"\{([^{}]*)\\over([^{}]*)\}")


def _over_to_frac(m: re.Match[str]) -> str:
    num, den = m.group(1).strip(), m.group(2).strip()
    return f"\\frac{{{num}}}{{{den}}}"


# Typographic space inside a number: the gold writes 1.45459\,66142, a reader may emit
# "1.45459 66142" or "1.4545966142". All three are the same value.
_DIGIT_GAP = re.compile(r"(?<=\d)\s+(?=\d)")
_EMPTY_GROUP = re.compile(r"\{\s*\}")
_WHITESPACE = re.compile(r"\s+")
# Markdown noise: Nougat emits headings and bold. `_` is NOT touched -- it is a subscript.
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)


def normalize_latex(text: str) -> str:
    """Canonicalise LaTeX/markdown so two spellings of the same formula compare equal.

    Unicode NFC first (so a precomposed and a combining form of the same glyph agree),
    then the typographic collapses documented above. Idempotent: normalising an already
    normalised string is a no-op, which is what lets callers normalise once and cache.
    """
    s = unicodedata.normalize("NFC", text)

    s = _MD_HEADING.sub(" ", s)
    s = s.replace("**", " ")
    s = _ENVIRONMENTS.sub(" ", s)
    s = _LABELS.sub(" ", s)
    s = _MATH_DELIMS.sub(" ", s)
    s = s.replace("$$", " ").replace("$", " ")

    # Fold equivalent command spellings before any argument rewriting below.
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    # `{A \over B}` -> `\frac{A}{B}`, innermost first (loop until no `\over` sits inside a
    # brace pair with no nested braces of its own -- see _OVER's comment above).
    prev = None
    while prev != s:
        prev = s
        s = _OVER.sub(_over_to_frac, s)
    s = _UPRIGHT.sub("\\\\mathrm{", s)
    s = _ELLIPSIS.sub("\\\\dots", s)
    # Deleted outright, not replaced by a space: these macros sit flush against the
    # delimiter they size (`\left(`), so a space would leave `(z+1 )` != `(z+1)`.
    s = _LEFT_RIGHT.sub("", s)
    s = _SPACING_MACROS.sub(" ", s)

    # Brace single-token arguments. Applied twice: `\frac12` produces braces that can expose
    # a further single-token script argument, and one pass would leave it unbraced.
    for _ in range(2):
        s = _FRAC_ARGS.sub(r"\\frac{\1}{\2}", s)
        s = _SCRIPT_ARG.sub(r"\1{\2}", s)

    s = _EMPTY_GROUP.sub("", s)
    s = _WHITESPACE.sub(" ", s)
    s = _DIGIT_GAP.sub("", s)
    return s.strip()


# --- character-level F1 -------------------------------------------------------------------


def _lcs_length(a: str, b: str) -> int:
    """Length of the longest common subsequence of two strings.

    Order-sensitive by construction, which is the whole point (see `ocr_f1`). Uses the
    standard two-row dynamic program, O(len(a)*len(b)) time but only O(min) memory.

    Common prefix and suffix are stripped first. That is exact for LCS -- some optimal
    alignment always matches a shared leading/trailing run greedily -- and it is what keeps
    the usual case (a mostly-correct transcription) fast instead of quadratic in the full
    page length.
    """
    if not a or not b:
        return 0

    start = 0
    limit = min(len(a), len(b))
    while start < limit and a[start] == b[start]:
        start += 1
    end = 0
    while end < (limit - start) and a[len(a) - 1 - end] == b[len(b) - 1 - end]:
        end += 1
    matched = start + end

    a_mid = a[start : len(a) - end]
    b_mid = b[start : len(b) - end]
    if not a_mid or not b_mid:
        return matched

    if len(a_mid) < len(b_mid):  # iterate over the longer string, keep the shorter row
        a_mid, b_mid = b_mid, a_mid

    prev = [0] * (len(b_mid) + 1)
    for ch_a in a_mid:
        cur = [0] * (len(b_mid) + 1)
        for j, ch_b in enumerate(b_mid, 1):
            cur[j] = prev[j - 1] + 1 if ch_a == ch_b else max(prev[j], cur[j - 1])
        prev = cur
    return matched + prev[-1]


def ocr_f1(pred: str, gold: str) -> float:
    """Character-level F1 between a predicted and a gold transcription, after normalisation.

    Both sides go through `normalize_latex` first, so the score measures reading accuracy
    rather than notation style. Overlap is the longest common *subsequence*, so
    precision = LCS/len(pred), recall = LCS/len(gold), F1 = their harmonic mean.

    Subsequence, not a bag of characters, because a bag is blind to order: a page whose
    regions come out in the wrong reading order contains exactly the same characters and
    would score a perfect 1.0. Measured on a real formula line, shuffling word order scores
    1.000 under a bag-of-characters F1 and 0.420 here -- and reading order is precisely what
    vision/layout.py is built to get right, so the metric has to be able to see it.

    Returns 0.0 if either side is empty after normalisation, 1.0 for identical input.
    """
    p = normalize_latex(pred)
    g = normalize_latex(gold)
    if not p or not g:
        return 0.0
    overlap = _lcs_length(p, g)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


# --- exact formula match ------------------------------------------------------------------

# A&S numbers its formulas 6.1.8, 9.1.10, ... That id is our citation anchor (summary.md 3f)
# and here it doubles as the alignment key: it tells us which predicted formula to compare
# against which gold formula, without needing to align the whole page first.
# Two spellings of the same thing must both parse, or this stops measuring OCR quality and
# starts measuring notation style -- the exact failure `normalize_latex` exists to prevent:
#
#   gold (hand-written, grading_kit/labels.jsonl):  6.1.1  \Gamma(z)=\int_0^\infty ...
#   Nougat (what the reader actually emits):        **6.1.1**: \(\Gamma(z)\!=\!\int ...\)
#
# The original pattern required the id at line start followed by whitespace, so it matched
# the gold and NOTHING the reader produces. Step 16's first Kaggle run therefore scored
# exact_formula_match = 0.0000 on all three gold pages while the transcripts demonstrably
# contained 6.1.1, 6.1.2 and 6.1.3 -- a metric that could not have returned anything but
# zero, whatever the reader did. Markdown emphasis around the id and a trailing colon are
# now optional; the id and a real separator are still required, so "6.1.1x" does not match.
_EMPH = r"(?:\*{1,2}|_{1,2})?"
# id and body on the SAME line: gold's "6.1.1  \Gamma..." and Nougat's "**6.1.1**: \(...\)"
_FORMULA_LINE = re.compile(rf"^\s*{_EMPH}\s*(\d+\.\d+\.\d+)\s*{_EMPH}\s*(?::\s*|\s+)(\S.*)$")
# id ALONE on its line, body in the following paragraph -- Nougat's other layout, seen on
# printed p.360: "**9.1.7**" / blank / "\(J_{\nu}(z)\sim...\)". A page can use both forms.
_FORMULA_ID_ONLY = re.compile(rf"^\s*{_EMPH}\s*(\d+\.\d+\.\d+)\s*{_EMPH}\s*:?\s*$")


def extract_formulas(text: str) -> dict[str, str]:
    """Map A&S formula id -> normalised formula body.

    Handles the three layouts this project actually encounters. All three must parse, or
    the metric measures which *notation* a reader chose rather than whether it read the
    maths correctly:

    1. gold, id then body on one line, continuations indented
       (grading_kit/labels.jsonl)::

           6.1.1  \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0)
                  =k^z\\int_0^\\infty t^{z-1}e^{-kt}\\,dt

    2. Nougat inline, emphasised id and a colon::

           **6.1.1**: \\(\\Gamma(z)\\!=\\!\\int_{0}^{\\infty}...\\)

    3. Nougat standalone, id on its own line and the body in the next paragraph::

           **9.1.7**

           \\(J_{\\nu}(z){\\,\\sim}({\\frac{1}{2}}z)^{\\nu}/\\Gamma(\\nu{+}1)\\)

    A prose line in column 0, or the next formula id, closes the current formula. If an id
    appears more than once the first occurrence wins -- later ones are usually
    cross-references, not restatements.
    """
    formulas: dict[str, str] = {}
    lines = text.splitlines()

    def store(fid: str, body: list[str]) -> None:
        normalised = normalize_latex(" ".join(body))
        if normalised and fid not in formulas:
            formulas[fid] = normalised

    def starts_formula(line: str) -> bool:
        return bool(_FORMULA_LINE.match(line) or _FORMULA_ID_ONLY.match(line))

    i, n = 0, len(lines)
    while i < n:
        inline = _FORMULA_LINE.match(lines[i])
        id_only = _FORMULA_ID_ONLY.match(lines[i])

        if inline:
            fid, body = inline.group(1), [inline.group(2)]
            i += 1
            # layout 1: indented continuation lines belong to this formula
            while (
                i < n
                and lines[i].strip()
                and lines[i][:1].isspace()
                and not starts_formula(lines[i])
            ):
                body.append(lines[i].strip())
                i += 1
            store(fid, body)
            continue

        if id_only:
            fid = id_only.group(1)
            i += 1
            while i < n and not lines[i].strip():  # layout 3: skip the blank separator
                i += 1
            body = []
            while i < n and lines[i].strip() and not starts_formula(lines[i]):
                body.append(lines[i].strip())
                i += 1
            store(fid, body)
            continue

        i += 1

    return formulas


def exact_formula_match(pred: str, gold: str) -> float:
    """Fraction of the gold page's numbered formulas that were read *exactly* right.

    Reported alongside `ocr_f1` because character F1 flatters mathematics (summary.md 4e):
    one wrong digit in a five-term expansion costs a single character but breaks the formula
    for anyone who uses it. This is the strict complement -- a formula counts only if its
    whole normalised body matches.

    Formulas are paired by their A&S id, so a missing or extra formula elsewhere on the page
    cannot shift the alignment. Returns 0.0 when the gold page has no numbered formulas
    (a pure prose or table page), so callers should weight by `len(extract_formulas(gold))`
    rather than averaging this across pages blindly.
    """
    gold_formulas = extract_formulas(gold)
    if not gold_formulas:
        return 0.0
    pred_formulas = extract_formulas(pred)
    hits = sum(1 for fid, body in gold_formulas.items() if pred_formulas.get(fid) == body)
    return hits / len(gold_formulas)


# --- A3 metrics: symbols are locked by tests/test_structure.py, bodies land in A3 ----------


def recall_at_k(retrieved: list, gold: list, k: int) -> float:
    raise NotImplementedError  # A3: retrieval quality


def groundedness(answer: Answer) -> float:
    raise NotImplementedError  # no-hallucination


def citation_accuracy(answer: Answer) -> float:
    raise NotImplementedError


def ece(confidences: Any, correct: Any) -> float:
    raise NotImplementedError  # calibration


def subgroup_gap(scores_by_group: dict) -> float:
    raise NotImplementedError  # fairness
