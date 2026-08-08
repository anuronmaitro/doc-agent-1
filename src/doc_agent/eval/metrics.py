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
_FORMULA_LINE = re.compile(r"^\s*(\d+\.\d+\.\d+)\s+(.*)$")


def extract_formulas(text: str) -> dict[str, str]:
    """Map A&S formula id -> normalised formula body.

    A formula starts at a line beginning with its id and continues through any following
    *indented* lines, which is how the gold writes multi-line formulas:

        6.1.1  \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0)
               =k^z\\int_0^\\infty t^{z-1}e^{-kt}\\,dt

    A blank line, or any line starting in column 0 that is not a new id (a prose heading
    like "Euler's Formula"), closes the current formula. If an id appears more than once,
    the first occurrence wins -- later ones are usually cross-references, not restatements.
    """
    formulas: dict[str, str] = {}
    current: str | None = None
    body: list[str] = []

    def close() -> None:
        if current is not None:
            normalised = normalize_latex(" ".join(body))
            if normalised and current not in formulas:
                formulas[current] = normalised

    for line in text.splitlines():
        match = _FORMULA_LINE.match(line)
        if match:
            close()
            current, body = match.group(1), [match.group(2)]
        elif current is not None and line.strip() and line[:1].isspace():
            body.append(line.strip())
        else:
            close()
            current, body = None, []
    close()
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
