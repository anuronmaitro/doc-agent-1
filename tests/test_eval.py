"""Unit test home for eval. IMPLEMENT — CI runs these."""

import random

import pytest

from doc_agent.eval.metrics import (
    exact_formula_match,
    extract_formulas,
    normalize_latex,
    ocr_f1,
)


def _brute_lcs(a: str, b: str) -> int:
    """Textbook full-matrix LCS, used only to cross-check the optimised implementation."""
    grid = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            grid[i][j] = grid[i - 1][j - 1] + 1 if ca == cb else max(grid[i - 1][j], grid[i][j - 1])
    return grid[-1][-1]


class TestNormalisation:
    @pytest.mark.parametrize(
        "a,b",
        [
            (r"\tfrac12", r"\frac{1}{2}"),
            (r"\dfrac{1}{2}", r"\frac{1}{2}"),
            (r"x^2", r"x^{2}"),
            (r"\int_0^\infty", r"\int_{0}^{\infty}"),
            (r"\left(z+1\right)", r"(z+1)"),
            (r"a\,b", r"a b"),
            (r"a\quad b", r"a  b"),
            (r"1.45459\,66142", r"1.45459 66142"),
            (r"\ldots", r"\dots"),
            (r"\cdots", r"\dots"),
            (r"\operatorname{Si}(x)", r"\mathrm{Si}(x)"),
            (r"\text{Ci}(x)", r"\mathrm{Ci}(x)"),
            (r"$\Gamma(z)$", r"\Gamma(z)"),
            (r"\[\Gamma(z)\]", r"\Gamma(z)"),
            (r"\begin{equation}\Gamma(z)\end{equation}", r"\Gamma(z)"),
            (r"\Gamma(z)\label{eq:1}", r"\Gamma(z)"),
            ("## Ascending Series", "Ascending Series"),
            ("**Euler**", "Euler"),
        ],
    )
    def test_equivalent_spellings_collapse(self, a, b):
        assert normalize_latex(a) == normalize_latex(b)

    @pytest.mark.parametrize(
        "a,b",
        [
            (r"x^2", r"x^3"),
            (r"\Gamma(z)", r"\Gamma(w)"),
            (r"\frac{1}{2}", r"\frac{1}{3}"),
            (r"J_0(z)", r"J_1(z)"),
            (r"1.45459", r"1.45458"),
            (r"\Re z>0", r"\Re z<0"),
        ],
    )
    def test_genuinely_different_maths_stays_different(self, a, b):
        """Normalisation must collapse typography, never mathematics."""
        assert normalize_latex(a) != normalize_latex(b)

    def test_is_idempotent(self):
        raw = r"\tfrac12 \left( \int_0^\infty t^{z-1}\,dt \right) \quad \ldots"
        once = normalize_latex(raw)
        assert normalize_latex(once) == once

    def test_nfc_unicode(self):
        assert normalize_latex("e\u0301") == normalize_latex("\u00e9")

    def test_subscript_underscore_is_not_stripped_as_markdown(self):
        assert "_" in normalize_latex(r"J_{0}(z)")


class TestOcrF1:
    def test_identical_scores_one(self):
        gold = r"6.1.8 \Gamma(\tfrac12)=\pi^{1/2}"
        assert ocr_f1(gold, gold) == 1.0

    def test_equivalent_notation_scores_one(self):
        """Different spelling of the same formula is a perfect read, not a penalty."""
        assert ocr_f1(r"\Gamma(\tfrac12)=\pi^{1/2}", r"\Gamma(\frac{1}{2})=\pi^{1/2}") == 1.0

    def test_near_miss_scores_high_but_below_one(self):
        gold = r"J_\nu(z)=(\tfrac12 z)^\nu/\Gamma(\nu+1)"
        pred = r"J_\nu(z)=(\tfrac12 z)^\nu/\Gamma(\nu+2)"  # one wrong character
        score = ocr_f1(pred, gold)
        assert 0.90 < score < 1.0

    def test_unrelated_text_scores_low(self):
        assert ocr_f1("completely unrelated prose", r"\Gamma(z)=\int_0^\infty") < 0.45

    def test_is_order_sensitive(self):
        """A bag-of-characters F1 would score a scrambled page 1.0; this must not."""
        gold = "9.1.10 J_v(z) equals the ascending series of Bessel functions"
        scrambled = " ".join(reversed(gold.split()))
        assert ocr_f1(scrambled, gold) < 0.75

    def test_empty_sides(self):
        assert ocr_f1("", "abc") == 0.0
        assert ocr_f1("abc", "") == 0.0

    def test_symmetric_in_the_f1_sense(self):
        a, b = r"\Gamma(z)=\int_0^\infty t^{z-1}", r"\Gamma(z)=\int_0^\infty"
        assert ocr_f1(a, b) == pytest.approx(ocr_f1(b, a))

    def test_partial_read_scores_between(self):
        gold = r"\Gamma(z)=\int_0^\infty t^{z-1}e^{-t}\,dt"
        half = r"\Gamma(z)=\int_0^\infty"
        assert 0.0 < ocr_f1(half, gold) < 1.0


class TestLcsOptimisation:
    """The prefix/suffix strip in _lcs_length is an exactness claim — verify it."""

    def test_matches_brute_force_on_random_strings(self):
        from doc_agent.eval.metrics import _lcs_length

        rng = random.Random(42)
        for _ in range(200):
            a = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 25)))
            b = "".join(rng.choice("abcd") for _ in range(rng.randint(0, 25)))
            assert _lcs_length(a, b) == _brute_lcs(a, b), (a, b)

    def test_matches_brute_force_with_shared_affixes(self):
        from doc_agent.eval.metrics import _lcs_length

        rng = random.Random(7)
        for _ in range(200):
            prefix = "".join(rng.choice("xy") for _ in range(rng.randint(0, 8)))
            suffix = "".join(rng.choice("xy") for _ in range(rng.randint(0, 8)))
            a = prefix + "".join(rng.choice("abc") for _ in range(rng.randint(0, 12))) + suffix
            b = prefix + "".join(rng.choice("abc") for _ in range(rng.randint(0, 12))) + suffix
            assert _lcs_length(a, b) == _brute_lcs(a, b), (a, b)


GOLD_PAGE = (
    "6. Gamma Function and Related Functions. Mathematical Properties.\n"
    "6.1. Gamma (Factorial) Function.\n"
    "Euler's Integral\n"
    "6.1.1  \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0)\n"
    "       =k^z\\int_0^\\infty t^{z-1}e^{-kt}\\,dt \\quad (\\Re z>0)\n"
    "Euler's Formula\n"
    "6.1.2  \\Gamma(z)=\\lim_{n\\to\\infty}\\frac{n!\\,n^{z}}{z(z+1)}\n"
    "6.1.8  \\Gamma(\\tfrac12)=\\pi^{1/2}\n"
)


class TestFormulaExtraction:
    def test_finds_every_numbered_formula(self):
        assert set(extract_formulas(GOLD_PAGE)) == {"6.1.1", "6.1.2", "6.1.8"}

    def test_indented_continuation_joins_its_formula(self):
        body = extract_formulas(GOLD_PAGE)["6.1.1"]
        assert "k^{z}" in body, "the indented second line belongs to 6.1.1"

    def test_prose_heading_closes_a_formula(self):
        """'Euler's Formula' sits in column 0, so it must not be swallowed into 6.1.1."""
        assert "Euler" not in extract_formulas(GOLD_PAGE)["6.1.1"]

    def test_page_with_no_numbered_formulas(self):
        assert extract_formulas("Just prose about Bessel functions.\n") == {}

    def test_first_occurrence_wins(self):
        text = "6.1.8  \\Gamma(\\tfrac12)=\\pi^{1/2}\n6.1.8  see above\n"
        assert "pi" in extract_formulas(text)["6.1.8"]


class TestExactFormulaMatch:
    def test_perfect_read_scores_one(self):
        assert exact_formula_match(GOLD_PAGE, GOLD_PAGE) == 1.0

    def test_notation_variant_still_counts_as_exact(self):
        pred = GOLD_PAGE.replace("\\tfrac12", "\\frac{1}{2}")
        assert exact_formula_match(pred, GOLD_PAGE) == 1.0

    def test_one_wrong_digit_loses_that_formula_entirely(self):
        """The point of this metric: a near-miss is not a partial credit."""
        pred = GOLD_PAGE.replace("\\pi^{1/2}", "\\pi^{1/3}")
        assert exact_formula_match(pred, GOLD_PAGE) == pytest.approx(2 / 3)

    def test_is_stricter_than_char_f1(self):
        pred = GOLD_PAGE.replace("\\pi^{1/2}", "\\pi^{1/3}")
        assert exact_formula_match(pred, GOLD_PAGE) < ocr_f1(pred, GOLD_PAGE)

    def test_missing_formula_counts_against(self):
        pred = GOLD_PAGE.replace("6.1.8  \\Gamma(\\tfrac12)=\\pi^{1/2}\n", "")
        assert exact_formula_match(pred, GOLD_PAGE) == pytest.approx(2 / 3)

    def test_extra_formula_does_not_shift_alignment(self):
        pred = GOLD_PAGE + "9.9.9  \\text{something else}\n"
        assert exact_formula_match(pred, GOLD_PAGE) == 1.0

    def test_gold_without_formulas_returns_zero(self):
        assert exact_formula_match("anything", "prose only, no numbered formulas") == 0.0
