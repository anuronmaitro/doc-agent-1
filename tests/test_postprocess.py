"""Unit test home for llm/postprocess.py. IMPLEMENT — CI runs these."""

from doc_agent import hooks
from doc_agent.contracts import Chunk
from doc_agent.eval import metrics
from doc_agent.llm import postprocess


def _chunk(id_: str, text: str, score: float = 0.5) -> Chunk:
    return Chunk(id=id_, doc_id="d0", text=text, page_ids=["p0"], score=score)


SYNTHESIZE_OK = """ANSWER: Gamma(1/2) = sqrt(pi)
CITATIONS: c1
RATIONALE: c1 states this directly, well ahead of any runner-up.
"""


class TestParseSynthesizeOutput:
    def test_parses_all_three_fields(self):
        text, cites, rationale = postprocess._parse_synthesize_output(SYNTHESIZE_OK)
        assert text == "Gamma(1/2) = sqrt(pi)"
        assert cites == ["c1"]
        assert "runner-up" in rationale

    def test_multiline_answer_field_is_captured_whole(self):
        raw = "ANSWER: line one\nline two\nCITATIONS: c1\nRATIONALE: r\n"
        text, cites, _ = postprocess._parse_synthesize_output(raw)
        assert text == "line one\nline two"
        assert cites == ["c1"]

    def test_comma_separated_citations_are_split_and_stripped(self):
        raw = "ANSWER: a\nCITATIONS: c1,  c2 ,c3\nRATIONALE: r\n"
        _, cites, _ = postprocess._parse_synthesize_output(raw)
        assert cites == ["c1", "c2", "c3"]

    def test_none_citations_becomes_empty_list(self):
        raw = "ANSWER: a\nCITATIONS: NONE\nRATIONALE: r\n"
        _, cites, _ = postprocess._parse_synthesize_output(raw)
        assert cites == []

    def test_missing_answer_field_is_treated_as_malformed_abstention(self):
        text, cites, reason = postprocess._parse_synthesize_output("garbled, no fields at all")
        assert text == postprocess.INSUFFICIENT_EVIDENCE
        assert cites == []
        assert "malformed" in reason


class TestConfidence:
    """Real signal, not a constant -- monotonic in each of the four named inputs."""

    def test_higher_top_score_is_never_less_confident(self):
        low = postprocess._confidence(0.1, 0.0, False, False)
        high = postprocess._confidence(0.9, 0.0, False, False)
        assert high > low

    def test_higher_score_gap_is_never_less_confident(self):
        low = postprocess._confidence(0.5, 0.0, False, False)
        high = postprocess._confidence(0.5, 0.9, False, False)
        assert high > low

    def test_calculator_verification_raises_confidence(self):
        without = postprocess._confidence(0.5, 0.2, False, False)
        with_calc = postprocess._confidence(0.5, 0.2, True, False)
        assert with_calc > without

    def test_a_retried_answer_is_less_confident_than_a_fresh_one(self):
        fresh = postprocess._confidence(0.5, 0.2, False, False)
        retried = postprocess._confidence(0.5, 0.2, False, True)
        assert retried < fresh

    def test_always_bounded_to_unit_interval(self):
        assert 0.0 <= postprocess._confidence(1.0, 1.0, True, False) <= 1.0
        assert 0.0 <= postprocess._confidence(0.0, 0.0, False, True) <= 1.0


class TestFormatAnswer:
    def test_real_answer_with_valid_citation_is_grounded(self):
        pool = [_chunk("c1", "Gamma(1/2)=sqrt(pi) is a classic identity.", score=0.9)]
        ans = postprocess.format_answer(SYNTHESIZE_OK, pool, top_score=0.9, score_gap=0.3)
        assert ans.grounded is True
        assert ans.citations == [postprocess.Citation(chunk_id="c1", span=(0, len(pool[0].text)))]
        assert "Gamma(1/2) = sqrt(pi)" in ans.text
        assert ans.confidence > 0.0

    def test_self_reported_insufficient_evidence_abstains(self):
        raw = "ANSWER: INSUFFICIENT EVIDENCE\nCITATIONS: NONE\nRATIONALE: no support in evidence.\n"
        ans = postprocess.format_answer(raw, [_chunk("c1", "unrelated")])
        assert ans.grounded is False
        assert ans.confidence == 0.0
        assert ans.citations == []
        assert ans.text == postprocess.INSUFFICIENT_EVIDENCE

    def test_citation_not_in_the_retrieved_pool_is_dropped_not_trusted(self):
        """A chunk id the model names but never actually retrieved must not ground anything
        (prompts.SYNTHESIZE's own Rule 3) -- with nothing left to cite, this abstains."""
        raw = "ANSWER: a real-sounding claim\nCITATIONS: never_retrieved_id\nRATIONALE: r\n"
        ans = postprocess.format_answer(raw, [_chunk("c1", "the only real chunk")])
        assert ans.grounded is False
        assert ans.citations == []
        assert ans.text == postprocess.INSUFFICIENT_EVIDENCE

    def test_malformed_output_abstains(self):
        ans = postprocess.format_answer("not the expected format at all", [_chunk("c1", "x")])
        assert ans.grounded is False
        assert ans.text == postprocess.INSUFFICIENT_EVIDENCE


class TestGroundHook:
    def setup_method(self):
        hooks.clear()
        postprocess.register(hooks)

    def teardown_method(self):
        hooks.clear()

    def _grounded_answer(self):
        pool = [_chunk("c1", "Gamma(1/2)=sqrt(pi).", score=0.9)]
        return postprocess.format_answer(SYNTHESIZE_OK, pool, top_score=0.9)

    def test_pre_synthesis_call_with_no_answer_key_is_a_no_op(self):
        ctx = hooks.run(hooks.BEFORE_ANSWER, {"state": {"query": "q", "obs": []}})
        assert "answer" not in ctx  # never raised, never invented an "answer" key

    def test_already_enforced_abstention_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(metrics, "groundedness", lambda ans: 0.0)
        abstained = postprocess.format_answer(
            "ANSWER: INSUFFICIENT EVIDENCE\nCITATIONS: NONE\nRATIONALE: r\n", []
        )
        ctx = hooks.run(hooks.BEFORE_ANSWER, {"state": {}, "answer": abstained})
        assert ctx["answer"] is abstained  # untouched -- nothing left to correct
        assert "grounding_complaint" not in ctx

    def test_low_real_groundedness_downgrades_and_adds_a_complaint(self, monkeypatch):
        monkeypatch.setattr(metrics, "groundedness", lambda ans: 0.1)
        ans = self._grounded_answer()
        ctx = hooks.run(hooks.BEFORE_ANSWER, {"state": {}, "answer": ans})
        assert ctx["answer"].grounded is False
        assert ctx["answer"].confidence == 0.0
        assert "grounding_complaint" in ctx and ctx["grounding_complaint"]

    def test_high_real_groundedness_passes_through_unchanged(self, monkeypatch):
        monkeypatch.setattr(metrics, "groundedness", lambda ans: 0.95)
        ans = self._grounded_answer()
        ctx = hooks.run(hooks.BEFORE_ANSWER, {"state": {}, "answer": ans})
        assert ctx["answer"].grounded is True
        assert "grounding_complaint" not in ctx

    def test_a_broken_groundedness_check_fails_closed(self, monkeypatch):
        def _boom(_ans):
            raise RuntimeError("index unavailable")

        monkeypatch.setattr(metrics, "groundedness", _boom)
        ans = self._grounded_answer()
        ctx = hooks.run(hooks.BEFORE_ANSWER, {"state": {}, "answer": ans})
        assert ctx["answer"].grounded is False  # never lets a failed check pass as grounded
