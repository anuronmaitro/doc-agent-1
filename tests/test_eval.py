"""Unit test home for eval. IMPLEMENT — CI runs these."""

import random

import pytest

from doc_agent.contracts import Answer, Chunk, Citation
from doc_agent.eval import metrics
from doc_agent.eval.metrics import (
    citation_accuracy,
    exact_formula_match,
    extract_formulas,
    groundedness,
    normalize_latex,
    ocr_f1,
    recall_at_k,
    subgroup_gap,
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
            # Step 18b: Nougat's other common fraction spelling, old-style TeX `\over`.
            (r"{1\over 2}", r"\tfrac12"),
            (r"{2\over\pi}", r"\frac{2}{\pi}"),
            (r"{(n-k-1)!\over k!}", r"\frac{(n-k-1)!}{k!}"),
            # Two independent (non-nested) \over fractions in the same string.
            (
                r"{1\over 2}z+{1\over 3}w",
                r"\frac{1}{2}z+\frac{1}{3}w",
            ),
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
            (r"{1\over 2}", r"{1\over 3}"),
        ],
    )
    def test_genuinely_different_maths_stays_different(self, a, b):
        """Normalisation must collapse typography, never mathematics."""
        assert normalize_latex(a) != normalize_latex(b)

    def test_is_idempotent(self):
        raw = r"\tfrac12 \left( \int_0^\infty t^{z-1}\,dt \right) \quad \ldots"
        once = normalize_latex(raw)
        assert normalize_latex(once) == once

    def test_is_idempotent_with_nested_over(self):
        raw = r"{-({{1\over 2}}z)^{-n}\over\pi}\sum_{k=0}^{n-1}{(n-k-1)!\over k!}"
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


def _chunk(id_, page_ids, text="irrelevant prose"):
    return Chunk(id=id_, doc_id="ch06_gamma", text=text, page_ids=page_ids)


class TestRecallAtK:
    def test_gold_page_found_via_top_k(self):
        retrieved = [_chunk("c1", ["as_p0255"]), _chunk("c2", ["as_p0999"])]
        assert recall_at_k(retrieved, gold=["as_p0255"], k=2) == 1.0

    def test_gold_page_outside_k_scores_zero(self):
        retrieved = [_chunk("c1", ["as_p0999"]), _chunk("c2", ["as_p0255"])]
        assert recall_at_k(retrieved, gold=["as_p0255"], k=1) == 0.0

    def test_multiple_gold_pages_partial_credit(self):
        retrieved = [_chunk("c1", ["as_p0255"]), _chunk("c2", ["as_p0999"])]
        assert recall_at_k(retrieved, gold=["as_p0255", "as_p0256"], k=2) == 0.5

    def test_page_found_via_a_different_chunk_than_expected_still_counts(self):
        """recall_at_k is page-id recall, not chunk-id recall (see docstring)."""
        retrieved = [_chunk("visual|as_p0255", ["as_p0255"])]
        assert recall_at_k(retrieved, gold=["as_p0255"], k=1) == 1.0

    def test_does_not_resort_just_slices_first_k(self):
        retrieved = [_chunk("c1", ["as_p0999"]), _chunk("c2", ["as_p0255"])]
        assert recall_at_k(retrieved, gold=["as_p0255"], k=1) == 0.0
        assert recall_at_k(retrieved, gold=["as_p0255"], k=2) == 1.0

    def test_empty_gold_or_zero_k(self):
        retrieved = [_chunk("c1", ["as_p0255"])]
        assert recall_at_k(retrieved, gold=[], k=5) == 0.0
        assert recall_at_k(retrieved, gold=["as_p0255"], k=0) == 0.0


REAL_EXCERPT = r"\Gamma(z+1)=z\Gamma(z)"
REAL_CHUNK = Chunk(
    id="ch06_gamma|as_p0255|r00",
    doc_id="ch06_gamma",
    text=f"6.1.15  Recurrence relation. {REAL_EXCERPT} for all z.",
    page_ids=["as_p0255"],
)


@pytest.fixture(autouse=True)
def _clear_chunk_lookup_cache(monkeypatch):
    """The real lookup is a lazily-cached module global -- reset it every test so one test's
    monkeypatch (or a prior real load) can never leak into the next."""
    monkeypatch.setattr(metrics, "_CHUNK_LOOKUP_CACHE", None)


def _patch_chunks(monkeypatch, *chunks):
    lookup = {c.id: c for c in chunks}
    monkeypatch.setattr(metrics, "_get_chunk_lookup", lambda: lookup)


class TestCitationAccuracy:
    def test_valid_citation_scores_one(self, monkeypatch):
        _patch_chunks(monkeypatch, REAL_CHUNK)
        span = (
            REAL_CHUNK.text.index(REAL_EXCERPT),
            REAL_CHUNK.text.index(REAL_EXCERPT) + len(REAL_EXCERPT),
        )
        answer = Answer(
            text="By the recurrence relation, " + REAL_EXCERPT,
            citations=[Citation(chunk_id=REAL_CHUNK.id, span=span)],
            grounded=True,
            confidence=0.9,
        )
        assert citation_accuracy(answer) == 1.0

    def test_invented_chunk_id_scores_zero(self, monkeypatch):
        """A citation pointing at a chunk_id that was never actually retrieved/indexed --
        the fabricated-citation case the plan calls out explicitly."""
        _patch_chunks(monkeypatch, REAL_CHUNK)
        answer = Answer(
            text="Some claim.",
            citations=[Citation(chunk_id="ch99_nonexistent|as_p9999|r00", span=(0, 5))],
            grounded=False,
            confidence=0.5,
        )
        assert citation_accuracy(answer) == 0.0

    def test_span_out_of_bounds_scores_zero(self, monkeypatch):
        _patch_chunks(monkeypatch, REAL_CHUNK)
        answer = Answer(
            text="Some claim.",
            citations=[Citation(chunk_id=REAL_CHUNK.id, span=(0, len(REAL_CHUNK.text) + 50))],
            grounded=False,
            confidence=0.5,
        )
        assert citation_accuracy(answer) == 0.0

    def test_mixed_citations_partial_credit(self, monkeypatch):
        _patch_chunks(monkeypatch, REAL_CHUNK)
        good_span = (
            REAL_CHUNK.text.index(REAL_EXCERPT),
            REAL_CHUNK.text.index(REAL_EXCERPT) + len(REAL_EXCERPT),
        )
        answer = Answer(
            text=REAL_EXCERPT,
            citations=[
                Citation(chunk_id=REAL_CHUNK.id, span=good_span),
                Citation(chunk_id="invented", span=(0, 3)),
            ],
            grounded=False,
            confidence=0.5,
        )
        assert citation_accuracy(answer) == 0.5

    def test_no_citations_scores_zero(self):
        answer = Answer(text="Unsupported claim.", citations=[], grounded=False, confidence=0.1)
        assert citation_accuracy(answer) == 0.0


class TestGroundedness:
    def test_real_chunk_real_span_supported_claim_scores_high(self, monkeypatch):
        _patch_chunks(monkeypatch, REAL_CHUNK)
        span = (
            REAL_CHUNK.text.index(REAL_EXCERPT),
            REAL_CHUNK.text.index(REAL_EXCERPT) + len(REAL_EXCERPT),
        )
        answer = Answer(
            text=f"By the recurrence relation, {REAL_EXCERPT}, valid for all z.",
            citations=[Citation(chunk_id=REAL_CHUNK.id, span=span)],
            grounded=True,
            confidence=0.9,
        )
        assert groundedness(answer) > 0.9

    def test_cites_real_chunk_but_states_something_it_does_not_say(self, monkeypatch):
        """The obvious attack the plan calls out: a real chunk, a valid span, and a claim
        the excerpt never actually makes -- citation_accuracy would pass this, groundedness
        must not."""
        _patch_chunks(monkeypatch, REAL_CHUNK)
        span = (
            REAL_CHUNK.text.index(REAL_EXCERPT),
            REAL_CHUNK.text.index(REAL_EXCERPT) + len(REAL_EXCERPT),
        )
        answer = Answer(
            text="The gamma function is always exactly equal to 42.",
            citations=[Citation(chunk_id=REAL_CHUNK.id, span=span)],
            grounded=True,  # the model *claims* it's grounded; the metric must not trust that
            confidence=0.9,
        )
        assert citation_accuracy(answer) == 1.0  # structurally, the citation is real
        assert groundedness(answer) < 0.2  # but the content is not supported

    def test_invented_citation_cannot_ground_anything(self, monkeypatch):
        _patch_chunks(monkeypatch, REAL_CHUNK)
        answer = Answer(
            text=REAL_EXCERPT,
            citations=[Citation(chunk_id="invented|nowhere|r00", span=(0, 5))],
            grounded=True,
            confidence=0.9,
        )
        assert groundedness(answer) == 0.0

    def test_notation_variant_still_counts_as_grounded(self, monkeypatch):
        """LaTeX normalisation applies here too -- \\tfrac vs \\frac must not look unsupported."""
        chunk = Chunk(
            id="c1",
            doc_id="ch06_gamma",
            text=r"6.1.8  \Gamma(\tfrac12)=\pi^{1/2}",
            page_ids=["as_p0255"],
        )
        _patch_chunks(monkeypatch, chunk)
        answer = Answer(
            text=r"We have \Gamma(\frac{1}{2})=\pi^{1/2} by definition.",
            citations=[Citation(chunk_id="c1", span=(7, len(chunk.text)))],
            grounded=True,
            confidence=0.9,
        )
        assert groundedness(answer) > 0.9

    def test_no_citations_scores_zero(self):
        answer = Answer(text="Unsupported claim.", citations=[], grounded=False, confidence=0.1)
        assert groundedness(answer) == 0.0


class TestSubgroupGap:
    def test_two_groups_returns_the_gap(self):
        assert subgroup_gap({"formula": 0.7, "prose": 0.9}) == pytest.approx(0.2)

    def test_three_groups_max_minus_min(self):
        assert subgroup_gap({"formula": 0.5, "prose": 0.9, "table": 0.6}) == pytest.approx(0.4)

    def test_identical_scores_zero_gap(self):
        assert subgroup_gap({"formula": 0.8, "prose": 0.8}) == 0.0

    def test_single_group_returns_zero(self):
        assert subgroup_gap({"formula": 0.8}) == 0.0

    def test_empty_returns_zero(self):
        assert subgroup_gap({}) == 0.0
