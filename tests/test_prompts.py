"""Unit test home for llm/prompts.py. IMPLEMENT — CI runs these."""

import pytest

from doc_agent.llm.prompts import DECIDE, JUDGE, SYNTHESIZE

# The nine locked tool names (tests/test_structure.py::test_tool_names_locked). DECIDE must
# only ever offer the READ/ACT subset -- retrieve/rerank are the code-driven evidence gate,
# never a prompt choice (plan_a3.md Step 6).
_RETRIEVAL_TOOLS = {"retrieve", "rerank"}
_ROUTABLE_TOOLS = {
    "read_page",
    "enhance_page",
    "extract",
    "aggregate",
    "cite",
    "calculator",
    "escalate_to_human",
}


class TestTemplatesAreFilledIn:
    @pytest.mark.parametrize("template", [DECIDE, SYNTHESIZE, JUDGE])
    def test_not_a_placeholder_stub(self, template):
        assert "IMPLEMENT" not in template

    @pytest.mark.parametrize("template", [DECIDE, SYNTHESIZE, JUDGE])
    def test_is_a_non_trivial_string(self, template):
        assert isinstance(template, str)
        assert len(template) > 100


class TestSynthesize:
    def _render(self, **overrides):
        args = {
            "query": "What is Gamma(1/2)?",
            "evidence": "[c1] (score=0.91) Gamma(1/2)=sqrt(pi)\n[c2] (score=0.40) unrelated",
        }
        args.update(overrides)
        return SYNTHESIZE.format(**args)

    def test_formats_with_query_and_evidence_placeholders(self):
        rendered = self._render()
        assert "What is Gamma(1/2)?" in rendered
        assert "Gamma(1/2)=sqrt(pi)" in rendered

    def test_forces_citations(self):
        assert "CITATIONS" in SYNTHESIZE
        assert "cite" in SYNTHESIZE.lower()

    def test_permits_abstention_as_first_class_output(self):
        """INSUFFICIENT EVIDENCE must be presented as a correct outcome, not merely allowed."""
        assert "INSUFFICIENT EVIDENCE" in SYNTHESIZE
        assert "not a failure" in SYNTHESIZE or "unpenalised" in SYNTHESIZE

    def test_requires_the_runner_up_rationale(self):
        assert "runner-up" in SYNTHESIZE
        assert "RATIONALE" in SYNTHESIZE

    def test_evidence_is_labelled_as_data_not_instructions(self):
        """Prompt-injection hygiene (plan_a3.md Step 6, item 4) -- must belong here."""
        lowered = SYNTHESIZE.lower()
        assert "never" in lowered and "instruction" in lowered
        assert "=== evidence" in lowered and "=== end evidence" in lowered

    def test_output_format_is_fully_specified(self):
        for field in ("ANSWER:", "CITATIONS:", "RATIONALE:"):
            assert field in SYNTHESIZE


class TestDecide:
    def test_formats_with_its_placeholders(self):
        rendered = DECIDE.format(
            query="q", evidence="[c1] (score=0.5) text", trace_so_far="(none yet)"
        )
        assert "q" in rendered
        assert "(none yet)" in rendered

    def test_only_offers_read_act_tools_never_retrieve_or_rerank(self):
        """The mandatory evidence-gate widen/re-search branch is code in decide(), not a
        prompt choice -- DECIDE must explicitly rule out retrieve/rerank as something to
        pick, and must never present either as an example choice."""
        for tool in _ROUTABLE_TOOLS:
            assert tool in DECIDE, f"{tool} should be offered as a choice"
        assert "do not choose retrieve" in DECIDE.lower()
        assert "TOOL: retrieve" not in DECIDE
        assert "TOOL: rerank" not in DECIDE

    def test_is_thin_relative_to_synthesize(self):
        """plan_a3.md Step 6: 'Keep DECIDE thin.'"""
        assert len(DECIDE) < len(SYNTHESIZE)

    def test_output_format_is_fully_specified(self):
        for field in ("TOOL:", "ARGS:", "WHY:"):
            assert field in DECIDE


class TestJudge:
    def _render(self, **overrides):
        args = {
            "query": "Does the recurrence hold for all z?",
            "evidence": "[c1] Gamma(z+1)=z*Gamma(z)",
            "answer": "Yes, by the recurrence relation shown in the evidence.",
        }
        args.update(overrides)
        return JUDGE.format(**args)

    def test_formats_with_its_placeholders(self):
        rendered = self._render()
        assert "Does the recurrence hold for all z?" in rendered
        assert "Yes, by the recurrence relation" in rendered

    def test_has_an_explicit_written_rubric(self):
        """plan_a3.md Step 6 + Sec.5: the rubric must be written here, not invented later
        while filling the form."""
        for criterion in ("CORRECTNESS", "COMPLETENESS", "GROUNDEDNESS"):
            assert criterion in JUDGE

    def test_scores_are_bounded_and_summed(self):
        assert "0, 1, or 2" in JUDGE
        assert "TOTAL" in JUDGE
        assert "/6" in JUDGE

    def test_correct_abstention_scores_full_marks(self):
        """A judge that penalises a correct 'insufficient evidence' would recreate exactly
        the fabrication pressure SYNTHESIZE is built to remove."""
        assert "insufficient evidence" in JUDGE.lower()
        assert "score 2" in JUDGE.lower() or "not a lesser answer" in JUDGE.lower()

    def test_evidence_is_labelled_as_data_not_instructions(self):
        assert "=== evidence" in JUDGE.lower()
        assert "never an instruction" in JUDGE.lower() or "not an instruction" in JUDGE.lower()

    def test_output_format_is_fully_specified(self):
        for field in ("CORRECTNESS:", "COMPLETENESS:", "GROUNDEDNESS:", "TOTAL:", "VERDICT:"):
            assert field in JUDGE


class TestNoStrayPromptStrings:
    """plan_a3.md Step 6 'Done when': no prompt strings anywhere else. A spot-check of the
    modules that will actually call these templates (Steps 9-11, 16) -- not a repo-wide
    heuristic scan, which would be too fragile to be a reliable gate."""

    @pytest.mark.parametrize(
        "module_path",
        [
            "src/doc_agent/agent/agent.py",
            "src/doc_agent/agent/tools.py",
            "src/doc_agent/llm/postprocess.py",
            "src/doc_agent/llm/client.py",
        ],
    )
    def test_module_does_not_import_or_redefine_prompt_bodies(self, module_path):
        text = open(module_path, encoding="utf-8").read()
        for banned in ("ANSWER:", "CITATIONS:", "RATIONALE:", "INSUFFICIENT EVIDENCE"):
            assert banned not in text, f"{module_path} appears to embed a prompt body"
