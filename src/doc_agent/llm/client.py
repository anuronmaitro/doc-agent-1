"""LLM — the single LLM call wrapper (all model calls go through here)"""

from __future__ import annotations

import time
from typing import Any

from groq import APIConnectionError, APITimeoutError, Groq, InternalServerError, RateLimitError

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from ..settings import settings

logger = get_logger(__name__)

# Transient failures worth retrying: a rate limit, a dropped connection, a timeout, or a
# 5xx on Groq's side. Deliberately does NOT include AuthenticationError, BadRequestError,
# etc. -- those are config problems retrying can't fix, and retrying one just burns the
# 30-requests/minute free-tier budget on calls that will never succeed (plan_a3.md D1).
_RETRYABLE: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0


class LLM:
    """Model set by cfg['agent']['model'] (Groq, openai/gpt-oss-120b -- DECISION D1,
    plan_a3.md §5). Key read through settings.settings.llm_api_key, never os.environ
    (settings.py is FIXED and says so)."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        if not settings.llm_api_key:
            raise RuntimeError(
                "LLM: no API key set. Put a real Groq key in LLM_API_KEY in .env "
                "(see .env.example) -- settings.py reads it from there, never os.environ."
            )
        self._client = Groq(api_key=settings.llm_api_key)
        # Step 13's guardrails read these to enforce cfg['agent']['budget_usd']; Step 22's
        # eval run reports them as the real, measured cost of the required Kaggle passes.
        self.call_count = 0
        self.total_tokens = 0

    def complete(self, prompt: str, **kw: Any) -> str:
        """One LLM call to cfg['agent']['model']. temperature=0 unless a caller overrides
        via kw -- deterministic by default, since Step 24's reproducibility claim and
        test_crosscutting.py::test_rerun_reproduces_metrics both depend on it. Retries with
        exponential backoff on the transient errors in _RETRYABLE; anything else (a bad key,
        a malformed request) raises immediately on the first attempt."""
        model = self.cfg["agent"]["model"]
        params: dict[str, Any] = {"temperature": 0, **kw}
        max_retries = int(params.pop("max_retries", DEFAULT_MAX_RETRIES))

        attempt = 0
        while True:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    **params,
                )
                break
            except _RETRYABLE as exc:
                if attempt >= max_retries:
                    logger.error(f"llm.client: giving up after {max_retries} retries: {exc}")
                    raise
                wait = DEFAULT_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    f"llm.client: {type(exc).__name__} on attempt {attempt + 1}/"
                    f"{max_retries + 1}, retrying in {wait:.1f}s"
                )
                time.sleep(wait)
                attempt += 1

        self.call_count += 1
        if response.usage is not None:
            self.total_tokens += response.usage.total_tokens
        text = response.choices[0].message.content
        logger.info(f"llm.client: call #{self.call_count} to {model} ({len(text or '')} chars)")
        return text or ""
