"""Unit test home for OCR.

Step 18b regression tests: the degeneracy detector (`_failure_reason`) is the piece of
`vision/ocr.py` that decides whether a transcribed page is kept or discarded, so a wrong
call here silently throws away good pages or keeps broken ones -- exactly the bug this
repair fixes. Covers plan.md Step 18b's explicit spec ("\\qquad x30 must be caught;
\\begin{tabular}{c c c c} must not be") plus the edge cases found while calibrating the
fix against the real 594-page Step 16 corpus (whitespace-fragile spirals, the bare "&"
false positive, tiny coincidental blips).
"""

from doc_agent.contracts import Region
from doc_agent.vision.ocr import (
    MIN_PAGE_CHARS,
    _failure_reason,
    _is_degenerate,
    _retry_page_by_region,
)


class TestFailureReasonMissingOrEmpty:
    def test_missing_page_marker_is_caught(self):
        text = "Some table header\n[MISSING_PAGE_POST]"
        assert _failure_reason(text) == "nougat-missing-page-marker"

    def test_near_empty_page_is_caught(self):
        assert _failure_reason("too short") == "empty-or-near-empty"
        assert len("too short") < MIN_PAGE_CHARS

    def test_ordinary_page_is_sound(self):
        text = "6.1.1  \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0)\n" * 4
        assert _failure_reason(text) is None
        assert _is_degenerate(text) is False


class TestFailureReasonRepetitionDegeneration:
    def test_qquad_x30_is_caught(self):
        """plan.md Step 18b's own regression spec: a 30-copy \\qquad spiral must be caught.

        The old detector (DEGEN_REPEAT_UNIT_MAX_LEN=4) could never match this -- \\qquad is
        6 characters, already longer than the unit length that could ever match.
        """
        text = "Preamble text before the spiral.\n" + "\\qquad" * 30
        assert _failure_reason(text) == "repetition-degeneration"

    def test_tabular_column_spec_is_not_caught(self):
        """plan.md Step 18b's own regression spec: a legitimate column spec must survive."""
        text = ("\\begin{tabular}{c c c c c c c c c c c c c c c c c c c c c c}\n" * 3) + (
            "0 & 1 & 2 & 3 \\\\\n" * 5
        )
        assert _failure_reason(text) != "repetition-degeneration"

    def test_pipe_delimited_column_spec_is_not_caught(self):
        text = "|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|\n" + ("data row\n" * 10)
        assert _failure_reason(text) != "repetition-degeneration"

    def test_whitespace_broken_spiral_is_still_caught(self):
        """A real Nougat spiral decodes with a stray space every ~13-14 copies (as_p0360,
        as_p0177), which breaks an exact-match backreference against the raw text. The
        detector must match on the whitespace-collapsed text instead."""
        unit = "\\qquad"
        copies = [unit] * 13 + [" " + unit] * 13 + [unit] * 13
        text = "".join(copies)
        assert _failure_reason(text) == "repetition-degeneration"

    def test_short_unit_needs_more_repeats_to_reach_the_span_floor(self):
        """A short unit repeated only enough times to clear the 13-repeat count but not
        the MIN_SPIRAL_SPAN_CHARS floor reads as a coincidental blip, not a stuck decoder
        (as_p0018/as_p0824: "\\," and "\\!" repeated ~13-14 times inside an otherwise-good
        page). Below the span floor it must NOT be flagged..."""
        short_spiral = "\\," * 13  # 26 chars: well under the 60-char span floor
        good_page = "6.1.1 \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0)\n" * 3
        text = good_page + short_spiral
        assert _failure_reason(text) != "repetition-degeneration"

    def test_short_unit_does_flag_once_it_clears_the_span_floor(self):
        """The same short unit DOES cross MIN_SPIRAL_SPAN_CHARS once repeated enough times
        to actually dominate a real page, rather than being a one-line blip."""
        long_spiral = "\\," * 40  # 80 chars: clears the 60-char floor
        text = "Some prose about the ascending series of Bessel functions.\n" + long_spiral
        assert _failure_reason(text) == "repetition-degeneration"

    def test_bare_ampersand_sparse_table_row_is_not_caught(self):
        """as_p0328: a sparse table row of bare "&" separators is legitimate LaTeX, not a
        spiral -- caught as a false positive before "&" was added to TABULAR_UNIT_RE."""
        text = ("Row label " + "& " * 20 + "\n") * 2
        assert _failure_reason(text) != "repetition-degeneration"

    def test_ampersand_mixed_with_math_content_is_still_caught(self):
        """as_p0856: a unit like "&\\(\\times\\)" contains "&" but also backslash/math
        content, so adding "&" to TABULAR_UNIT_RE must not create a new blind spot for
        genuine spirals that merely happen to include a table separator."""
        text = "Preamble.\n" + "&\\(\\times\\)" * 13
        assert _failure_reason(text) == "repetition-degeneration"

    def test_mid_page_spiral_is_caught_not_just_tail(self):
        """The whole-page scan (added in Step 18b) must catch a spiral that starts mid-page
        and is followed by clean, non-repeating text -- the old tail-only check would miss
        this because the last DEGEN_NGRAM*DEGEN_MIN_REPEATS tokens are not the spiral."""
        text = (
            "Clean opening paragraph about Bessel functions of the first kind.\n"
            + "\\qquad" * 25
            + "\nA clean closing paragraph that does not repeat anything at all.\n"
        )
        assert _failure_reason(text) == "repetition-degeneration"

    def test_original_tail_check_still_fires(self):
        """The pre-existing word-level tail check (independent of the character-unit regex
        above) must keep working for the failure mode it was built for."""
        pattern = " ".join(f"tok{i}" for i in range(12))
        text = "Normal opening text. " + (pattern + " ") * 4
        assert _failure_reason(text) == "repetition-degeneration"

    def test_step28_real_spiral_longer_than_old_20char_cap_is_now_caught(self):
        """Real regression: the fine-tuned reader's validation run (Step 28, curve point
        n=122) produced `as_p0334`'s worst-scoring prediction (char-F1 0.067) as a spiral
        on the unit "-\\mu xP_{\\tau}^{n}(z) " -- 22 characters, past the OLD
        DEGEN_REPEAT_UNIT_MAX_LEN=20 cap, so it was scored as a low-quality success instead
        of a failure. Reproduced verbatim (short prose lead-in + the real repeated unit)."""
        unit = "-\\mu xP_{\\tau}^{n}(z) "
        assert len(unit) > 20  # the exact old-threshold miss this regression guards
        text = "(z^2-1)!P_{\\tau}^{n-1}(z) " + unit * 15
        assert _failure_reason(text) == "repetition-degeneration"

    def test_legitimate_varying_table_rows_are_not_caught_by_wider_unit_cap(self):
        """Guard against the 20->60 widening creating a new false positive: real table
        rows are long-ish and share structure, but their actual digits differ row to row,
        so no fixed unit can repeat consecutively -- must stay unflagged."""
        rows = "\n".join(f"{i}.0  {i}.0000  {i}.1234  {i}.5678  {i}.9012" for i in range(20))
        assert _failure_reason(rows + "\n" * 2 + rows.replace("0.0", "1.1")) is None


class TestFailureReasonBlockRepetition:
    def test_duplicate_display_equation_block_is_caught(self):
        """Step 21 finding 1: `as_p0441`-style failure -- a display equation block repeats
        once, verbatim, later on the page, with different content in between. The
        consecutive-unit spiral check above cannot see this (the blocks aren't adjacent)."""
        block = "\\(J_{\\nu}(z)\\sim(\\tfrac12 z)^{\\nu}/\\Gamma(\\nu+1)\\quad(z\\to0)\\)"
        assert len(block) >= 60
        text = (
            "9.1.7\n\n"
            + block
            + "\n\nSome unrelated intervening prose about convergence.\n\n"
            + block
        )
        assert _failure_reason(text) == "block-repetition-degeneration"

    def test_short_duplicate_block_under_the_floor_is_not_caught(self):
        """A short, incidentally-repeated block (e.g. a one-line section header appearing
        twice, legitimately) must not trip the check -- only blocks >= MIN_BLOCK_DUP_CHARS."""
        text = "Notation\n\nSome real content paragraph here that is not repeated.\n\nNotation"
        assert _failure_reason(text) != "block-repetition-degeneration"

    def test_distinct_blocks_are_not_caught(self):
        """A normal multi-paragraph page (every block different) must not be flagged."""
        text = "\n\n".join(
            f"Paragraph {i} discusses a distinct topic in enough length to clear any floor."
            for i in range(5)
        )
        assert _failure_reason(text) is None


class _FakeReader:
    """Duck-typed stand-in for Reader: `_retry_page_by_region` only ever calls
    `_generate_region`, so a real model/GPU is not needed to test its recombination and
    failure-propagation logic."""

    def __init__(self, texts: dict[tuple[int, int, int, int], str]) -> None:
        self._texts = texts

    def _generate_region(self, region: Region) -> tuple[str, float]:
        return self._texts[region.bbox], 0.9


class _CrashingReader:
    """Duck-typed stand-in that raises on specific regions, simulating the real crash
    (`ValueError: axes don't match array`) found on the full-book run -- proves the retry
    loop's per-region try/except keeps the other regions' results, rather than propagating
    the exception and losing the whole page (or the whole run)."""

    def __init__(
        self,
        crashes_on: set[tuple[int, int, int, int]],
        texts: dict[tuple[int, int, int, int], str],
    ) -> None:
        self._crashes_on = crashes_on
        self._texts = texts

    def _generate_region(self, region: Region) -> tuple[str, float]:
        if region.bbox in self._crashes_on:
            raise ValueError("axes don't match array")
        return self._texts[region.bbox], 0.9


class TestRetryPageByRegion:
    def _regions(self, n: int) -> list[Region]:
        return [Region(page_id="as_p0001", bbox=(0, i, 10, i + 1), kind="text") for i in range(n)]

    def test_recombines_sound_regions_into_success(self):
        regions = self._regions(2)
        text_a = "6.1.1 \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0). " * 3
        text_b = (
            "6.1.2 \\Gamma(z)=\\lim_{n\\to\\infty}\\frac{n!\\,n^{z}}{z(z+1)(z+2)\\cdots(z+n)}. " * 3
        )
        reader = _FakeReader({(0, 0, 10, 1): text_a, (0, 1, 10, 2): text_b})
        result = _retry_page_by_region(reader, regions)
        assert result is not None
        recombined, region_texts, region_confs = result
        assert "6.1.1" in recombined and "6.1.2" in recombined
        assert region_texts == [text_a, text_b]
        assert region_confs == [0.9, 0.9]

    def test_empty_regions_are_dropped_from_the_recombination(self):
        regions = self._regions(2)
        text_a = "6.1.1 \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0). " * 3
        reader = _FakeReader({(0, 0, 10, 1): text_a, (0, 1, 10, 2): ""})
        result = _retry_page_by_region(reader, regions)
        assert result is not None
        recombined, _texts, _confs = result
        assert recombined == text_a

    def test_still_degenerate_recombination_returns_none(self):
        """If retrying region-by-region does not actually fix anything, the caller must
        keep the original failure rather than silently accepting a still-broken page."""
        regions = self._regions(2)
        reader = _FakeReader(
            {
                (0, 0, 10, 1): "\\qquad" * 30,
                (0, 1, 10, 2): "",
            }
        )
        assert _retry_page_by_region(reader, regions) is None

    def test_a_crashing_region_does_not_crash_the_whole_page(self):
        """Step 18b defect 5: a full-book run died outright when one region's crop crashed
        Nougat's own preprocessing (ValueError: axes don't match array, on a degenerate
        crop). One bad region among ~1040 pages must not cost the whole multi-hour job --
        it's treated as empty text, and the OTHER regions on the page still make it through."""
        regions = self._regions(2)
        good_text = "6.1.1 \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0). " * 3
        reader = _CrashingReader(crashes_on={(0, 0, 10, 1)}, texts={(0, 1, 10, 2): good_text})
        result = _retry_page_by_region(reader, regions)
        assert result is not None
        recombined, region_texts, region_confs = result
        assert recombined == good_text
        assert region_texts == ["", good_text]
        assert region_confs == [0.0, 0.9]


class TestGenerateRegionCropGuard:
    """Step 18b defect 5: the actual root cause behind the crash above. A degenerate crop
    (near-zero width or height) must never reach the model at all -- these tests use a real
    on-disk image so `_page_image_path` -> `Image.crop` runs for real, and prove the model
    call itself is never reached for a bad bbox but still happens for a normal one."""

    def _make_reader_with_page(self, tmp_path, monkeypatch):
        from PIL import Image as PILImage

        import doc_agent.vision.ocr as ocr_mod

        PILImage.new("RGB", (200, 200), "white").save(tmp_path / "as_p0001.png")
        monkeypatch.setattr(ocr_mod, "INTERIM_DIR", tmp_path)
        monkeypatch.setattr(ocr_mod, "PAGES_DIR", tmp_path)
        return ocr_mod

    def test_degenerate_crop_skips_the_model_call(self, tmp_path, monkeypatch):
        ocr_mod = self._make_reader_with_page(tmp_path, monkeypatch)

        def _boom(self, image):
            raise AssertionError("the model must not be called on a degenerate crop")

        monkeypatch.setattr(ocr_mod.Reader, "_generate", _boom)
        reader = ocr_mod.Reader({"ocr": {}})

        degenerate = Region(page_id="as_p0001", bbox=(0, 0, 50, 1), kind="text")
        assert reader._generate_region(degenerate) == ("", 0.0)

    def test_reasonably_sized_crop_still_reaches_the_model(self, tmp_path, monkeypatch):
        ocr_mod = self._make_reader_with_page(tmp_path, monkeypatch)
        monkeypatch.setattr(ocr_mod.Reader, "_generate", lambda self, image: ("ok", 0.7))
        reader = ocr_mod.Reader({"ocr": {}})

        sane = Region(page_id="as_p0001", bbox=(0, 0, 50, 50), kind="text")
        assert reader._generate_region(sane) == ("ok", 0.7)
