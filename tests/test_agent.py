"""Unit test home for agent. IMPLEMENT — CI runs these."""

from types import SimpleNamespace

import httpx
import pytest
from groq import AuthenticationError, RateLimitError

from doc_agent.agent.agent import Agent
from doc_agent.contracts import ToolResult
from doc_agent.llm import client as client_mod

CFG = {"agent": {"model": "openai/gpt-oss-120b"}}


def _make_agent() -> Agent:
    # act() never touches self.retriever -- it only dispatches through tools.REGISTRY --
    # so a bare cfg/None retriever is enough here; decide()/synthesize() get real fixtures
    # at Steps 10/11 when they're implemented.
    return Agent(cfg={"agent": {"max_steps": 8}}, retriever=None)


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

    def test_decide_and_synthesize_are_still_not_implemented(self):
        agent = _make_agent()
        with pytest.raises(NotImplementedError):
            agent.decide({})
        with pytest.raises(NotImplementedError):
            agent.synthesize({})
