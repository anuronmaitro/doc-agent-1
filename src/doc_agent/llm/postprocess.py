"""LLM — answer post-process / format / abstention"""

from __future__ import annotations

import re
from typing import Any

from ..contracts import *  # noqa
from ..eval import metrics
from ..logging_conf import get_logger
from .prompts import SYNTHESIZE_ABSTAIN_TEXT as INSUFFICIENT_EVIDENCE
from .prompts import SYNTHESIZE_FIELDS

logger = get_logger(__name__)

# A per-answer LIVE gate, deliberately looser than eval/metrics.py's own reported headline
# target (>= 0.90, an AGGREGATE bar over many answers). This threshold only needs to catch
# an answer with essentially no real support before it reaches a user -- the aggregate bar
# is eval/metrics.py's own job to report, not this gate's.
GROUNDEDNESS_THRESHOLD = 0.5

# Field regex built FROM prompts.SYNTHESIZE_FIELDS at import time, not retyped here --
# tests/test_prompts.py::TestNoStrayPromptStrings bans this file from embedding its own copy
# of the field labels, so the pattern only ever contains whatever prompts.py itself exports.
_FIELD_PATTERN = "|".join(re.escape(f) for f in SYNTHESIZE_FIELDS)
_FIELD_RE = re.compile(
    rf"^({_FIELD_PATTERN}):\s*(.*?)(?=\n(?:{_FIELD_PATTERN}):|\Z)", re.DOTALL | re.MULTILINE
)


def _parse_synthesize_output(raw: str) -> tuple[str, list[str], str]:
    """Split the SYNTHESIZE template's fixed multi-field output (prompts.SYNTHESIZE_FIELDS,
    in order) into its parts. Missing the first (answer) field entirely means the model
    didn't follow the format -- fail closed as an abstention rather than guessing which part
    of a malformed reply is the answer."""
    fields = {m.group(1): m.group(2).strip() for m in _FIELD_RE.finditer(raw)}
    answer_field, citations_field, rationale_field = SYNTHESIZE_FIELDS

    if answer_field not in fields:
        return INSUFFICIENT_EVIDENCE, [], f"malformed model output (missing {answer_field} field)"
    answer_text = fields[answer_field]

    citations_raw = fields.get(citations_field, "NONE")
    citation_ids = (
        []
        if not citations_raw or citations_raw.upper() == "NONE"
        else [cid.strip() for cid in citations_raw.split(",") if cid.strip()]
    )

    rationale = fields.get(rationale_field, "")
    return answer_text, citation_ids, rationale


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _confidence(
    top_score: float, score_gap: float, calculator_verified: bool, retried: bool
) -> float:
    """A real signal, not a constant -- Step 21 calibrates this and our second NFR (ECE <=
    0.05) is meaningless against a value that never varies. Four inputs, exactly the ones
    Step 11's own ORDER named: retrieval strength (top_score), how much stronger the best
    chunk was than the runner-up (score_gap -- the same top1-minus-top2 formula
    agent/tools.py's Rerank tool already uses, Step 7), whether the calculator independently
    verified a numeric claim (a real, mechanical check, worth a fixed bonus), and whether
    this answer only passed after a verify-and-correct retry (weaker evidence than a clean
    first pass, even though it ended up grounded -- a fixed penalty, not disqualifying)."""
    conf = 0.6 * _clamp01(top_score) + 0.2 * _clamp01(score_gap)
    conf += 0.1 if calculator_verified else 0.0
    conf += 0.1 if not retried else -0.1
    return _clamp01(conf)


def format_answer(
    raw: str,
    citations: list,
    *,
    top_score: float = 0.0,
    score_gap: float = 0.0,
    calculator_verified: bool = False,
    retried: bool = False,
) -> Answer:
    """Parse `raw` (prompts.SYNTHESIZE's fixed ANSWER/CITATIONS/RATIONALE output) and build
    an `Answer`. `citations` is the pool of `Chunk`s actually retrieved this turn (state's
    "chunks") -- a cited id resolves to `Citation(chunk_id, span=(0, len(chunk.text)))` only
    if it is literally in that pool; any other id (hallucinated, or real but never actually
    retrieved) is silently dropped, matching the SYNTHESIZE prompt's own Rule 3 ("never a
    chunk id you have not actually seen"). Enforces abstention: a self-reported
    INSUFFICIENT_EVIDENCE, or zero citations surviving resolution, both return the same
    ungrounded, zero-confidence, uncited Answer -- an answer with nothing real to cite
    cannot be grounded regardless of what its text claims.

    This is the STRUCTURAL half of the grounding contract (parse + resolve + enforce a
    citation actually exists). The DEEPER semantic check -- do the citations actually
    *support* the answer's claims -- is `_ground`'s job at the BEFORE_ANSWER hook below,
    which can still downgrade a `grounded=True` result from here."""
    answer_text, citation_ids, rationale = _parse_synthesize_output(raw)
    chunk_by_id = {c.id: c for c in citations}
    resolved = [
        Citation(chunk_id=cid, span=(0, len(chunk_by_id[cid].text)))
        for cid in citation_ids
        if cid in chunk_by_id
    ]

    if answer_text == INSUFFICIENT_EVIDENCE or not resolved:
        return Answer(text=INSUFFICIENT_EVIDENCE, citations=[], grounded=False, confidence=0.0)

    confidence = _confidence(top_score, score_gap, calculator_verified, retried)
    text = f"{answer_text}\n\nRationale: {rationale}" if rationale else answer_text
    return Answer(text=text, citations=resolved, grounded=True, confidence=confidence)


def register(hooks: Any) -> None:
    """Wire the grounding / abstention gate at BEFORE_ANSWER.

    Fires from two different call sites with two different ctx shapes, and must never raise
    from either:
      - run()'s own single BEFORE_ANSWER call, `{"state": state}` -- no "answer" key yet
        (synthesize() hasn't run). Nothing to check yet; a no-op pass-through.
      - synthesize()'s own calls (Step 11), `{"state": state, "answer": ans}`, once per
        synthesis attempt (first pass, and the one verify-and-correct retry if needed). This
        is where the real check happens: an already-enforced abstention from format_answer()
        (`grounded=False`) is left alone -- nothing left to correct -- but a `grounded=True`
        answer gets the deeper `eval.metrics.groundedness` check, and is downgraded (with a
        `grounding_complaint` added to ctx for synthesize()'s retry) if its citations don't
        actually support what it says.
    """

    def _ground(ctx: dict) -> dict:
        answer = ctx.get("answer")
        if answer is None:
            return ctx
        try:
            if not answer.grounded:
                return ctx
            score = metrics.groundedness(answer)
            if score < GROUNDEDNESS_THRESHOLD:
                ctx["answer"] = answer.model_copy(update={"grounded": False, "confidence": 0.0})
                ctx["grounding_complaint"] = (
                    f"the previous answer's citations do not actually support its claims "
                    f"(groundedness={score:.2f}, need >= {GROUNDEDNESS_THRESHOLD}); "
                    f"re-answer using ONLY what the cited evidence literally says, or say "
                    f"{INSUFFICIENT_EVIDENCE} if it truly does not support one"
                )
        except Exception:
            # Fail closed: a broken grounding check must not let an unverified answer
            # through silently -- treat it exactly like a failed check, not a passed one.
            logger.warning(
                "postprocess._ground: groundedness check failed; treating as unsupported"
            )
            ctx["answer"] = answer.model_copy(update={"grounded": False, "confidence": 0.0})
            ctx["grounding_complaint"] = "the grounding check itself failed; re-answer carefully"
        return ctx

    hooks.register(hooks.BEFORE_ANSWER, _ground)
