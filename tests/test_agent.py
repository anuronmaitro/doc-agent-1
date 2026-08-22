"""Unit test home for agent. IMPLEMENT — CI runs these."""

from types import SimpleNamespace

import httpx
import pytest
from groq import AuthenticationError, RateLimitError

from doc_agent.agent.agent import Agent
from doc_agent.contracts import Chunk, ToolResult
from doc_agent.llm import client as client_mod

CFG = {"agent": {"model": "openai/gpt-oss-120b"}}


def _make_agent() -> Agent:
    # act() never touches self.retriever -- it only dispatches through tools.REGISTRY --
    # so a bare cfg/None retriever is enough here; synthesize() gets a real fixture at
    # Step 11 when it's implemented.
    return Agent(cfg={"agent": {"max_steps": 8}}, retriever=None)


# decide()'s own cfg -- small k/k_step/k_max so a widening test takes 2-3 calls, not 4
# (real configs/config.yaml uses k=10/k_step=10/k_max=40; the *shape* of the policy is
# what's under test here, not the real numbers).
DECIDE_RETRIEVE_CFG = {"k": 1, "k_step": 1, "k_max": 3, "weak_threshold": 0.5}


class _StubRetriever:
    """Controllable stand-in for retrieval.retriever.Retriever -- returns one canned
    top_score per call, in order, so a test can script an exact weak/strong sequence
    without a real index or encoder. Mirrors this file's own _FakeCompletions pattern."""

    def __init__(self, top_scores: list) -> None:
        self._top_scores = list(top_scores)
        self.calls: list = []

    def retrieve(self, query: str, k: int) -> list:
        self.calls.append((query, k))
        score = self._top_scores.pop(0)
        return [Chunk(id="c0", doc_id="d0", text="chunk text", page_ids=["p0"], score=score)]


def _agent_with_stub(top_scores: list):
    stub = _StubRetriever(top_scores)
    agent = Agent(cfg={"agent": {"max_steps": 8}, "retrieve": DECIDE_RETRIEVE_CFG}, retriever=stub)
    return agent, stub


_REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _fake_response(text: str = "answer", total_tokens: int = 42) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def _rate_limit_error() -> RateLimitError:
    return RateLimitError("rate limited", response=httpx.Response(429, request=_REQ), body=None)


def _auth_error() -> AuthenticationError:
    return AuthenticationError("bad key", response=httpx.Response(401, request=_REQ), body=None)


class _FakeCompletions:
    """Pops one canned response/exception per call, in order -- lets a test script
    exactly the sequence of failures-then-success it wants to exercise."""

    def __init__(self, side_effects: list) -> None:
        self._side_effects = list(side_effects)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        effect = self._side_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


class _FakeGroqClient:
    def __init__(self, side_effects: list) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(side_effects))


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Every test here gets a fake key regardless of the real .env -- CI has no key,
    and a test that needs one is a test that fails on a grader's clean clone."""
    monkeypatch.setattr(client_mod.settings, "llm_api_key", "fake-test-key")


def _make_llm(monkeypatch, side_effects: list) -> client_mod.LLM:
    monkeypatch.setattr(client_mod, "Groq", lambda api_key: _FakeGroqClient(side_effects))
    return client_mod.LLM(CFG)


class TestLLMClient:
    def test_missing_key_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(client_mod.settings, "llm_api_key", "")
        with pytest.raises(RuntimeError, match="LLM_API_KEY"):
            client_mod.LLM(CFG)

    def test_successful_call_returns_text(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_fake_response("the answer")])
        assert llm.complete("a question") == "the answer"

    def test_temperature_zero_by_default(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_fake_response()])
        llm.complete("q")
        assert llm._client.chat.completions.calls[0]["temperature"] == 0

    def test_temperature_override_is_respected(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_fake_response()])
        llm.complete("q", temperature=0.7)
        assert llm._client.chat.completions.calls[0]["temperature"] == 0.7

    def test_model_comes_from_config_not_hardcoded(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_fake_response()])
        llm.complete("q")
        assert llm._client.chat.completions.calls[0]["model"] == "openai/gpt-oss-120b"

    def test_call_count_and_token_count_accumulate(self, monkeypatch):
        llm = _make_llm(
            monkeypatch, [_fake_response(total_tokens=10), _fake_response(total_tokens=15)]
        )
        llm.complete("q1")
        llm.complete("q2")
        assert llm.call_count == 2
        assert llm.total_tokens == 25

    def test_retries_on_rate_limit_then_succeeds(self, monkeypatch):
        llm = _make_llm(monkeypatch, [_rate_limit_error(), _fake_response("recovered")])
        result = llm.complete("q", max_retries=3)
        assert result == "recovered"
        assert len(llm._client.chat.completions.calls) == 2  # one failure, one success

    def test_gives_up_after_max_retries_and_raises(self, monkeypatch):
        llm = _make_llm(
            monkeypatch,
            [_rate_limit_error(), _rate_limit_error(), _rate_limit_error()],
        )
        with pytest.raises(RateLimitError):
            llm.complete("q", max_retries=2)
        # 1 initial attempt + 2 retries = 3 calls total, then it gives up
        assert len(llm._client.chat.completions.calls) == 3

    def test_auth_error_is_not_retried(self, monkeypatch):
        """A bad key is a config problem, not a transient one -- retrying it would just
        burn the free-tier rate-limit budget on calls that can never succeed."""
        llm = _make_llm(monkeypatch, [_auth_error(), _fake_response("should never be reached")])
        with pytest.raises(AuthenticationError):
            llm.complete("q", max_retries=3)
        assert len(llm._client.chat.completions.calls) == 1  # no retry attempted

    def test_max_retries_kwarg_is_not_forwarded_to_the_api_call(self, monkeypatch):
        """max_retries is this wrapper's own control knob -- Groq's API has no such
        parameter, so it must be popped before the kwargs reach chat.completions.create."""
        llm = _make_llm(monkeypatch, [_fake_response()])
        llm.complete("q", max_retries=1)
        assert "max_retries" not in llm._client.chat.completions.calls[0]


class TestAgentAct:
    """Step 9: act() only -- registry dispatch. decide()/synthesize() land at Steps 10/11."""

    def test_dispatches_to_registered_tool_by_name(self):
        agent = _make_agent()
        result = agent.act({"tool": "calculator", "args": {"expr": "2 + 2"}})
        assert isinstance(result, ToolResult)
        assert result.ok is True
        assert result.payload["value"] == 4

    def test_unknown_tool_returns_ok_false_not_raise(self):
        agent = _make_agent()
        result = agent.act({"tool": "not_a_real_tool", "args": {}})
        assert result.ok is False
        assert "not_a_real_tool" in result.payload["reason"]

    def test_missing_args_key_defaults_to_empty_kwargs(self):
        # aggregate's "count" op accepts an empty list, so a tool with no required args
        # exercises the action.get("args", {}) default without needing a real payload.
        agent = _make_agent()
        result = agent.act({"tool": "aggregate", "args": {"op": "count", "items": []}})
        assert result.ok is True
        assert result.payload["value"] == 0

    def test_synthesize_is_still_not_implemented(self):
        # decide() is implemented as of Step 10 -- only synthesize() (Step 11) still raises.
        agent = _make_agent()
        with pytest.raises(NotImplementedError):
            agent.synthesize({})


class TestAgentDecide:
    """Step 10: evidence-gated re-search -- the mandatory agentic behaviour.

    The trace, for each case below (walked through in the PR description, per the ORDER):
    each retrieval attempt decide() makes appends one {"top_score": ..., "k": ...} entry to
    state["obs"] BEFORE either widening or stopping -- so state["obs"] after decide() returns
    is the full widen-and-recheck trail, in order, with the branch on the real number visible
    directly (a strong score right away = one entry and done; weak scores show k climbing
    entry by entry until either a strong score appears or k_max is hit and state["abstain"]
    flips true). Step 12 (not this step) is what later turns this into traces/run.jsonl.
    """

    def test_strong_evidence_is_a_single_pass_no_widening(self):
        agent, stub = _agent_with_stub([0.9])
        state = {"query": "q", "obs": []}
        action = agent.decide(state)

        assert action == {"tool": "stop", "args": {}}
        assert [k for _, k in stub.calls] == [1]
        assert state["obs"] == [{"top_score": 0.9, "k": 1}]
        assert state["abstain"] is False

    def test_weak_then_strong_widens_exactly_once(self):
        agent, stub = _agent_with_stub([0.2, 0.9])
        state = {"query": "q", "obs": []}
        agent.decide(state)

        assert [k for _, k in stub.calls] == [1, 2]  # k + k_step, once
        assert state["obs"] == [
            {"top_score": 0.2, "k": 1},
            {"top_score": 0.9, "k": 2},
        ]
        assert state["abstain"] is False

    def test_weak_all_the_way_widens_to_k_max_then_abstains(self):
        agent, stub = _agent_with_stub([0.1, 0.1, 0.1])
        state = {"query": "q", "obs": []}
        action = agent.decide(state)

        assert [k for _, k in stub.calls] == [1, 2, 3]  # climbs to k_max, never further
        assert state["obs"] == [
            {"top_score": 0.1, "k": 1},
            {"top_score": 0.1, "k": 2},
            {"top_score": 0.1, "k": 3},
        ]
        assert state["abstain"] is True
        assert state["abstain_reason"] == "insufficient evidence"
        assert action == {"tool": "stop", "args": {}}  # never fabricates, never loops forever

    def test_k_never_exceeds_k_max(self):
        agent, stub = _agent_with_stub([0.1, 0.1, 0.1])
        state = {"query": "q", "obs": []}
        agent.decide(state)

        assert all(k <= DECIDE_RETRIEVE_CFG["k_max"] for _, k in stub.calls)
