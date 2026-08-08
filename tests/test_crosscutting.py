"""Cross-cutting features must work END TO END, not just exist in one file.
Un-skip and implement alongside the feature. CI runs these."""

import pytest


@pytest.mark.skip(reason="implement with grounding")
def test_grounding_unsupported_query_abstains():
    """An answer with no supporting evidence must abstain, not fabricate."""
    assert True


@pytest.mark.skip(reason="implement with security")
def test_injection_in_document_does_not_hijack():
    """A document containing 'ignore your instructions' must not change agent behaviour."""
    assert True


def test_pii_never_leaks_to_answer_or_log():
    """PII in the corpus must not appear in answers or logs."""
    from doc_agent import hooks
    from doc_agent.contracts import Chunk, ToolResult
    from doc_agent.governance import pii

    hooks.clear()
    pii.register(hooks)
    try:
        chunk = Chunk(
            id="c1",
            doc_id="ch01",
            text="Contact editor Milton Abramowitz at m.abramowitz@nbs.gov",
            page_ids=["as_p0001"],
        )
        ctx = hooks.run(hooks.AFTER_OCR, {"chunks": [chunk]})
        assert "Milton Abramowitz" not in ctx["chunks"][0].text
        assert "m.abramowitz@nbs.gov" not in ctx["chunks"][0].text

        obs = ToolResult(ok=True, payload={"snippet": "See Dr. John Todd for details."})
        state = {"query": "What is the gamma function?", "obs": [obs]}
        ctx = hooks.run(hooks.BEFORE_ANSWER, {"state": state})
        assert "Dr. John Todd" not in ctx["state"]["obs"][0].payload["snippet"]

        log_ctx = hooks.run(hooks.ON_LOG, {"message": "user jane.doe@example.com logged in"})
        assert "jane.doe@example.com" not in log_ctx["message"]
    finally:
        hooks.clear()


@pytest.mark.skip(reason="implement with tracing")
def test_trace_covers_every_step():
    """Every agent step and tool call must appear in the audit trail."""
    assert True


@pytest.mark.skip(reason="implement with reproducibility")
def test_rerun_reproduces_metrics():
    """A seeded re-run reproduces reported metrics within tolerance."""
    assert True
