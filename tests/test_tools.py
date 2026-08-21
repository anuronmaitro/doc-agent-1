import math

import pytest

from doc_agent.agent import hitl, tools
from doc_agent.contracts import Chunk, ToolResult


def test_registry_is_tool_subclasses():
    for t in tools.REGISTRY:
        assert issubclass(t, tools.Tool) and isinstance(t.name, str)


@pytest.fixture(autouse=True)
def _clear_module_caches(monkeypatch):
    """Both lazy caches are module globals -- reset before every test so one test's
    monkeypatch (or a prior real load) can never leak into the next."""
    monkeypatch.setattr(tools, "_CHUNK_LOOKUP_CACHE", None)
    monkeypatch.setattr(tools, "_RETRIEVER_CACHE", None)


def _chunk(id_, text, page_ids=None, score=0.0):
    return Chunk(
        id=id_, doc_id="ch06_gamma", text=text, page_ids=page_ids or ["as_p0255"], score=score
    )


def _patch_chunks(monkeypatch, *chunks):
    lookup = {c.id: c for c in chunks}
    monkeypatch.setattr(tools, "_get_chunk_lookup", lambda: lookup)


class _FakeRetriever:
    def __init__(self, chunks, exc=None):
        self._chunks = chunks
        self._exc = exc
        self.calls = []

    def retrieve(self, query, k=10):
        self.calls.append((query, k))
        if self._exc is not None:
            raise self._exc
        return self._chunks


class TestRetrieve:
    def test_empty_query_rejected_without_raising(self):
        result = tools.Retrieve()(query="   ")
        assert result.ok is False
        assert "reason" in result.payload

    def test_success_payload_shape(self, monkeypatch):
        chunks = [_chunk("c1", "text one", score=0.9), _chunk("c2", "text two", score=0.4)]
        monkeypatch.setattr(tools, "_get_retriever", lambda: _FakeRetriever(chunks))
        result = tools.Retrieve()(query="gamma function", k=5)
        assert result.ok is True
        assert result.payload == {"chunk_ids": ["c1", "c2"], "top_score": 0.9, "k": 5}

    def test_retriever_exception_becomes_ok_false_not_a_raise(self, monkeypatch):
        monkeypatch.setattr(
            tools, "_get_retriever", lambda: _FakeRetriever([], exc=RuntimeError("boom"))
        )
        result = tools.Retrieve()(query="q")
        assert result.ok is False
        assert "boom" in result.payload["reason"]

    def test_empty_result_gives_zero_top_score(self, monkeypatch):
        monkeypatch.setattr(tools, "_get_retriever", lambda: _FakeRetriever([]))
        result = tools.Retrieve()(query="q")
        assert result.ok is True
        assert result.payload["top_score"] == 0.0


def _fake_rerank(query, candidates, cfg):
    # Reverses input order and assigns descending scores -- enough to prove the tool used
    # this function's real return value rather than the input order.
    reordered = list(reversed(candidates))
    for i, c in enumerate(reordered):
        reordered[i] = c.model_copy(update={"score": 1.0 - i * 0.3})
    return reordered


class TestRerank:
    def test_empty_candidates_short_circuits(self):
        result = tools.Rerank()(query="q", candidates=[])
        assert result == ToolResult(ok=True, payload={"chunk_ids": [], "score_gap": 0.0})

    def test_accepts_chunk_objects_directly(self, monkeypatch):
        monkeypatch.setattr(tools.rerank_mod, "rerank", _fake_rerank)
        candidates = [_chunk("a", "x"), _chunk("b", "y")]
        result = tools.Rerank()(query="q", candidates=candidates)
        assert result.ok is True
        assert result.payload["chunk_ids"] == ["b", "a"]
        assert result.payload["score_gap"] == pytest.approx(0.3)

    def test_accepts_chunk_ids_and_resolves_them(self, monkeypatch):
        a, b = _chunk("a", "x"), _chunk("b", "y")
        _patch_chunks(monkeypatch, a, b)
        monkeypatch.setattr(tools.rerank_mod, "rerank", _fake_rerank)
        result = tools.Rerank()(query="q", candidates=["a", "b"])
        assert result.ok is True
        assert result.payload["chunk_ids"] == ["b", "a"]

    def test_unknown_chunk_id_rejected_without_raising(self, monkeypatch):
        _patch_chunks(monkeypatch)
        result = tools.Rerank()(query="q", candidates=["nonexistent"])
        assert result.ok is False
        assert "nonexistent" in result.payload["reason"]

    def test_single_candidate_has_zero_gap(self, monkeypatch):
        monkeypatch.setattr(tools.rerank_mod, "rerank", lambda q, c, cfg: c)
        result = tools.Rerank()(query="q", candidates=[_chunk("a", "x", score=0.7)])
        assert result.payload["score_gap"] == 0.0

    def test_rerank_exception_becomes_ok_false(self, monkeypatch):
        def _boom(q, c, cfg):
            raise RuntimeError("rerank exploded")

        monkeypatch.setattr(tools.rerank_mod, "rerank", _boom)
        result = tools.Rerank()(query="q", candidates=[_chunk("a", "x")])
        assert result.ok is False
        assert "rerank exploded" in result.payload["reason"]


class TestReadPage:
    def test_reads_a_real_ocr_page(self):
        result = tools.ReadPage()(page_id="as_p0255")
        assert result.ok is True
        assert result.payload["page_id"] == "as_p0255"
        assert "Gamma" in result.payload["text"]

    def test_missing_page_is_ok_false_not_a_raise(self):
        result = tools.ReadPage()(page_id="as_p9999_does_not_exist")
        assert result.ok is False
        assert "reason" in result.payload

    def test_empty_page_id_rejected(self):
        result = tools.ReadPage()(page_id="")
        assert result.ok is False


class TestEnhancePage:
    def test_disabled_in_config_is_honest_no_op(self, monkeypatch):
        monkeypatch.setattr(tools.config, "load", lambda: {"enhance": {"enabled": False}})
        result = tools.EnhancePage()(page_id="as_p0255")
        assert result.ok is True
        assert result.payload["enhanced"] is False
        assert "enhance.enabled=false" in result.payload["reason"]

    def test_enabled_in_config_still_does_not_fake_work(self, monkeypatch):
        monkeypatch.setattr(tools.config, "load", lambda: {"enhance": {"enabled": True}})
        result = tools.EnhancePage()(page_id="as_p0255")
        assert result.ok is True
        assert result.payload["enhanced"] is False

    def test_empty_page_id_rejected(self):
        result = tools.EnhancePage()(page_id="")
        assert result.ok is False


FORMULA_CHUNK_TEXT = (
    "6.1.1  \\Gamma(z)=\\int_0^\\infty t^{z-1}e^{-t}\\,dt \\quad (\\Re z>0)\n"
    "Some table: the value at z=1 is 1.0000000\n"
)


class TestExtract:
    def test_formula_id_uses_extract_formulas(self, monkeypatch):
        chunk = _chunk("c1", FORMULA_CHUNK_TEXT)
        _patch_chunks(monkeypatch, chunk)
        result = tools.Extract()(field="6.1.1", chunk_id="c1")
        assert result.ok is True
        assert "\\Gamma(z)" in result.payload["value"]
        start, end = result.payload["span"]
        assert chunk.text[start:end] == "6.1.1"

    def test_generic_field_falls_back_to_substring_search(self, monkeypatch):
        chunk = _chunk("c1", FORMULA_CHUNK_TEXT)
        _patch_chunks(monkeypatch, chunk)
        result = tools.Extract()(field="1.0000000", chunk_id="c1")
        assert result.ok is True
        assert result.payload["value"] == "1.0000000"

    def test_field_not_found_is_ok_false(self, monkeypatch):
        chunk = _chunk("c1", FORMULA_CHUNK_TEXT)
        _patch_chunks(monkeypatch, chunk)
        result = tools.Extract()(field="9.9.9", chunk_id="c1")
        assert result.ok is False

    def test_unknown_chunk_is_ok_false(self, monkeypatch):
        _patch_chunks(monkeypatch)
        result = tools.Extract()(field="anything", chunk_id="nope")
        assert result.ok is False


class TestAggregate:
    def test_count(self):
        assert tools.Aggregate()(op="count", items=["a", "b", "c"]).payload["value"] == 3

    def test_concat(self):
        result = tools.Aggregate()(op="concat", items=["a", "b"])
        assert result.payload["value"] == "a b"

    def test_sum(self):
        assert tools.Aggregate()(op="sum", items=[1, 2, 3]).payload["value"] == 6

    def test_mean(self):
        assert tools.Aggregate()(op="mean", items=[2, 4]).payload["value"] == 3

    def test_max_min(self):
        assert tools.Aggregate()(op="max", items=[1, 5, 3]).payload["value"] == 5
        assert tools.Aggregate()(op="min", items=[1, 5, 3]).payload["value"] == 1

    def test_unknown_op_is_ok_false(self):
        result = tools.Aggregate()(op="frobnicate", items=[1, 2])
        assert result.ok is False

    def test_empty_items_for_numeric_op_is_ok_false_not_a_raise(self):
        result = tools.Aggregate()(op="sum", items=[])
        assert result.ok is False

    def test_non_numeric_items_for_numeric_op_is_ok_false_not_a_raise(self):
        result = tools.Aggregate()(op="sum", items=["not", "numbers"])
        assert result.ok is False


class TestCite:
    def test_valid_span_builds_citation_and_rationale(self, monkeypatch):
        chunk = _chunk("c1", "the gamma function satisfies a recurrence", score=0.83)
        _patch_chunks(monkeypatch, chunk)
        result = tools.Cite()(chunk_id="c1", span=(4, 9))
        assert result.ok is True
        assert result.payload["citation"] == {"chunk_id": "c1", "span": (4, 9)}
        assert "c1" in result.payload["rationale"]
        assert "0.830" in result.payload["rationale"]

    def test_unknown_chunk_is_ok_false(self, monkeypatch):
        _patch_chunks(monkeypatch)
        result = tools.Cite()(chunk_id="nope", span=(0, 3))
        assert result.ok is False

    def test_out_of_bounds_span_is_ok_false(self, monkeypatch):
        chunk = _chunk("c1", "short")
        _patch_chunks(monkeypatch, chunk)
        result = tools.Cite()(chunk_id="c1", span=(0, 999))
        assert result.ok is False

    def test_malformed_span_is_ok_false_not_a_raise(self, monkeypatch):
        chunk = _chunk("c1", "short")
        _patch_chunks(monkeypatch, chunk)
        result = tools.Cite()(chunk_id="c1", span="not-a-tuple")
        assert result.ok is False

    def test_backwards_span_is_ok_false(self, monkeypatch):
        chunk = _chunk("c1", "short text")
        _patch_chunks(monkeypatch, chunk)
        result = tools.Cite()(chunk_id="c1", span=(5, 2))
        assert result.ok is False


class TestCalculator:
    def test_a1_worked_example_gamma_one_half(self):
        result = tools.Calculator()(expr="gamma(0.5)")
        assert result.ok is True
        assert result.payload["value"] == pytest.approx(1.77245385, abs=1e-6)

    def test_sqrt_pi_matches_the_same_example(self):
        result = tools.Calculator()(expr="sqrt(pi)")
        assert result.payload["value"] == pytest.approx(math.sqrt(math.pi))

    def test_basic_arithmetic(self):
        assert tools.Calculator()(expr="2 + 3 * 4").payload["value"] == 14
        assert tools.Calculator()(expr="(2 + 3) * 4").payload["value"] == 20
        assert tools.Calculator()(expr="2 ** 10").payload["value"] == 1024
        assert tools.Calculator()(expr="-5 + 2").payload["value"] == -3

    def test_division_by_zero_is_ok_false_not_a_raise(self):
        result = tools.Calculator()(expr="1 / 0")
        assert result.ok is False

    def test_syntax_error_is_ok_false_not_a_raise(self):
        result = tools.Calculator()(expr="2 + * 3")
        assert result.ok is False

    @pytest.mark.parametrize(
        "malicious",
        [
            "__import__('os').system('echo pwned')",
            "().__class__.__bases__[0]",
            "open('/etc/passwd').read()",
            "[x for x in range(10)]",
            "x",  # undefined name, not in the constant whitelist
            "1; 2",
        ],
    )
    def test_injection_attempts_are_rejected_without_raising(self, malicious):
        result = tools.Calculator()(expr=malicious)
        assert result.ok is False
        assert "reason" in result.payload

    def test_never_raises_for_any_string_input(self):
        """The defining requirement: garbage input must never propagate an exception."""
        for garbage in ["", "   ", ")(", "1//", "@#$%", "None", "True", "'a' + 'b'"]:
            result = tools.Calculator()(expr=garbage)
            assert isinstance(result, ToolResult)
            assert result.ok is False


class TestEscalateToHuman:
    def test_unimplemented_hitl_becomes_ok_false_not_a_raise(self, monkeypatch):
        """hitl.escalate is Step 13's job and currently raises -- this tool's own
        never-raise contract must hold regardless."""
        result = tools.EscalateToHuman()(reason="guardrail tripped", context={})
        assert result.ok is False
        assert "reason" in result.payload

    def test_passes_through_a_real_hitl_result_once_implemented(self, monkeypatch):
        expected = ToolResult(ok=True, payload={"queued": True, "ticket": "abc123"})
        monkeypatch.setattr(hitl, "escalate", lambda reason, context: expected)
        result = tools.EscalateToHuman()(reason="ambiguous evidence", context={"q": "x"})
        assert result == expected

    def test_hitl_raising_anything_becomes_ok_false(self, monkeypatch):
        def _boom(reason, context):
            raise ValueError("queue unavailable")

        monkeypatch.setattr(hitl, "escalate", _boom)
        result = tools.EscalateToHuman()(reason="r", context={})
        assert result.ok is False
        assert "queue unavailable" in result.payload["reason"]
