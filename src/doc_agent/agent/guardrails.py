"""Stage 6 — SECURITY — autonomy, budgets, prompt-injection defense"""

from __future__ import annotations

from typing import Any

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from . import hitl

logger = get_logger(__name__)


class GuardrailViolationError(Exception):
    """A genuine guardrail violation (budget / autonomy / injection) -- distinct from any
    other exception, so a caller or a test can tell "the guardrail blocked this on purpose"
    apart from an unrelated bug."""


# A1's own committed autonomy level (plan_a3.md Sec.5). The guardrail's job is to catch a
# config drift away from this, not to interpret what other autonomy levels would mean.
ALLOWED_AUTONOMY = "act-then-log"

# A1's promised HITL trigger: calibrated confidence below this after re-search to k_max.
# "Calibrated" is aspirational here -- Step 21 is the calibration step -- this constant and
# the check that uses it (agent.py's synthesize(), the k_max-abstain path) are the real,
# live trigger MECHANISM, using confidence exactly as Step 11 computes it today.
ESCALATION_CONFIDENCE_THRESHOLD = 0.50

# Deliberately NOT a real Groq price -- Groq's free tier is $0/call (D1, plan_a3.md Sec.5:
# "budget_usd becomes a non-binding ceiling, not a real constraint"). Kept as a real,
# testable formula rather than a permanent no-op so a future switch to a paid provider
# wouldn't silently lose budget enforcement.
ASSUMED_USD_PER_1K_TOKENS = 0.0005

# A1's own two named phrases (plan_a3.md Step 13's Do list) -- not a longer, fancier list.
# Honest framing, per check()'s own docstring: the REAL defence is Step 6's evidence-block
# prompt structure, which never lets retrieved text act as an instruction regardless of its
# content, no matter what this list does or doesn't catch. This is a second, best-effort
# detection layer -- catch it, log it, escalate it -- not the thing actually keeping the
# agent safe.
INJECTION_PATTERNS = ("ignore your instructions", "disregard the above")


def _find_injection(value: Any) -> str | None:
    """Recursively walk an action's args for a known injection phrase (case-insensitive
    substring match) -- same recursive-walk shape as governance/pii.py's _redact_in_place,
    since an action's args can nest dicts/lists/strings depending on which tool it targets."""
    if isinstance(value, str):
        lowered = value.lower()
        for pattern in INJECTION_PATTERNS:
            if pattern in lowered:
                return pattern
        return None
    if isinstance(value, dict):
        for v in value.values():
            found = _find_injection(v)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for v in value:
            found = _find_injection(v)
            if found:
                return found
        return None
    return None


class Guardrails:
    """Enforce autonomy level, step/cost budget, and instruction/content isolation."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["agent"]

    def reset(self) -> None:
        self.spent = 0.0
        self.steps = 0

    def check(self, action: dict, llm: Any | None = None) -> None:
        """Raise GuardrailViolationError on a genuine violation; otherwise return None -- an
        ordinary action must sail through with no exception at all.

        `llm` is optional. ON_TOOL_CALL's own ctx carries only `action` (`register()` below
        calls `g.check(ctx["action"])`, unchanged), so under the CURRENT wiring this is
        always called with `llm=None` and `self.spent` stays 0.0 -- honest, given budget_usd
        is a non-binding ceiling today (see ASSUMED_USD_PER_1K_TOKENS above). The formula
        itself is real and directly testable (`check(action, llm=<real LLM>)`), ready for
        whenever tracked spend actually needs to gate something.
        """
        self.steps += 1
        if self.steps > self.cfg["max_steps"]:
            raise GuardrailViolationError(
                f"max_steps exceeded: {self.steps} > {self.cfg['max_steps']} (A1's own cap)"
            )

        if llm is not None:
            self.spent = (llm.total_tokens / 1000) * ASSUMED_USD_PER_1K_TOKENS
            if self.spent > self.cfg["budget_usd"]:
                raise GuardrailViolationError(
                    f"budget_usd exceeded: ${self.spent:.4f} > ${self.cfg['budget_usd']}"
                )

        autonomy = self.cfg.get("autonomy")
        if autonomy != ALLOWED_AUTONOMY:
            raise GuardrailViolationError(
                f"disallowed autonomy level {autonomy!r}; A1 committed to "
                f"{ALLOWED_AUTONOMY!r} only"
            )

        matched = _find_injection(action)
        if matched:
            logger.warning(f"guardrails: possible prompt injection detected: {matched!r}")
            hitl.escalate(
                "prompt injection detected in tool-call content",
                {"action": action, "matched_pattern": matched},
            )
            raise GuardrailViolationError(
                f"refusing action: possible prompt injection detected ({matched!r}) -- "
                "escalated for human review, not treated as an instruction"
            )


def register(hooks: Any, cfg: dict) -> None:
    """Wire guardrails into every tool call."""
    g = Guardrails(cfg)
    g.reset()

    def _check(ctx: dict) -> dict:
        g.check(ctx["action"])  # budgets / autonomy / injection
        return ctx

    hooks.register(hooks.ON_TOOL_CALL, _check)
