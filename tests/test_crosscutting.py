"""Cross-cutting features must work END TO END, not just exist in one file.
Un-skip and implement alongside the feature. CI runs these."""

from types import SimpleNamespace

import pytest


def test_grounding_unsupported_query_abstains(monkeypatch):
    """An answer with no supporting evidence must abstain, not fabricate -- end to end
    through the real Agent.synthesize() + the real BEFORE_ANSWER _ground hook, covering the
    one verify-and-correct retry (D6) and the final abstain once it's exhausted."""
    from doc_agent import hooks
    from doc_agent.agent.agent import Agent
    from doc_agent.contracts import Chunk
    from doc_agent.eval import metrics
    from doc_agent.llm import client as client_mod
    from doc_agent.llm import postprocess

    hooks.clear()
    postprocess.register(hooks)
    try:
        # Every attempt is judged unsupported by the deep, index-backed check -- this
        # exercises the retry AND the final abstain, not just a lucky first pass.
        monkeypatch.setattr(metrics, "groundedness", lambda ans: 0.0)

        def _fake_response(text: str) -> SimpleNamespace:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                usage=SimpleNamespace(total_tokens=10),
            )

        class _FakeCompletions:
            def __init__(self, texts: list) -> None:
                self._texts = list(texts)
                self.calls: list[dict] = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return _fake_response(self._texts.pop(0))

        class _FakeGroqClient:
            def __init__(self, texts: list) -> None:
                self.chat = SimpleNamespace(completions=_FakeCompletions(texts))

        fake = _FakeGroqClient(
            [
                "ANSWER: the gamma function equals 42 here\nCITATIONS: c1\nRATIONALE: it just does.\n",
                "ANSWER: still an unsupported guess\nCITATIONS: c1\nRATIONALE: still guessing.\n",
            ]
        )
        monkeypatch.setattr(client_mod, "Groq", lambda api_key: fake)
        monkeypatch.setattr(client_mod.settings, "llm_api_key", "fake-test-key")

        chunk = Chunk(
            id="c1", doc_id="d0", text="unrelated real evidence text", page_ids=["p0"], score=0.9
        )
        agent = Agent(
            cfg={"agent": {"max_steps": 8, "model": "openai/gpt-oss-120b"}}, retriever=None
        )
        state = {
            "query": "What is Gamma(1/2)?",
            "obs": [],
            "chunks": [chunk],
            "abstain": False,
        }

        ans = agent.synthesize(state)

        assert ans.grounded is False
        assert ans.citations == []
        assert ans.text == postprocess.INSUFFICIENT_EVIDENCE  # never fabricates
        assert len(fake.chat.completions.calls) == 2  # exactly one verify-and-correct retry
    finally:
        hooks.clear()


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
