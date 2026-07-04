"""
Corrupted-answer regression tests for the faithfulness safety gate.

Happy-path pipeline tests never exercise the gate's actual job: rejecting a
hallucination. These cases feed the judge deliberately-broken answers (a
fabricated figure, an invented name, outside knowledge, a projection stated as
fact) and assert it REJECTS them — proving the gate still catches something and
hasn't degraded into a rubber stamp after a model/prompt change. The positive
case guards the opposite failure: over-rejecting a valid grounded answer.

Keep the corruptions blatant: the judge is only prompt-deterministic (the
Claude 5 family has deprecated `temperature`), so subtle edge cases would make
these flaky. We test "does the gate work at all", not "does it catch every
subtle case".
"""
import os

import pytest

from evals.faithfulness_eval import evaluate_faithfulness

# A fixed, self-contained context — the "source documents" for these cases.
# Figures match the known NovaTech Q4 2024 ground truth.
_CONTEXT = """[Source 1 — novatech_q4_2024.pdf, page 3]
Total Revenue for Q4 2024 was $1.42B, up from $1.31B in Q3 2024.

[Source 2 — novatech_q4_2024.pdf, page 7]
Departmental headcount, Q4 2024:
  R&D: 2,620 | Sales: 1,900 | Operations: 1,750 | G&A: 980
"""


@pytest.mark.integration
class TestFaithfulnessGate:
    """The judge hits the real claude-sonnet-5 model; requires ANTHROPIC_API_KEY."""

    def setup_method(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

    # ---- Positive: a grounded answer must PASS (guards against over-rejection) ----
    async def test_grounded_answer_is_accepted(self):
        answer = "Q4 2024 revenue was $1.42B, and R&D was the largest department at 2,620."
        result = await evaluate_faithfulness(answer, _CONTEXT, mode="standard")
        assert result.faithful is True
        assert result.unsupported_claims == []

    # ---- Negative: fabricated FIGURE must be REJECTED (the dangerous failure) ----
    async def test_fabricated_number_is_rejected(self):
        # $1.87B never appears in the context — the real figure is $1.42B.
        answer = "Q4 2024 revenue was $1.87B."
        result = await evaluate_faithfulness(answer, _CONTEXT, mode="standard")
        assert result.faithful is False
        assert result.unsupported_claims  # judge must say *why*

    # ---- Negative: invented PROPER NOUN must be rejected ----
    async def test_fabricated_name_is_rejected(self):
        answer = "The report was prepared by CFO Jane Halloran."  # no such name in context
        result = await evaluate_faithfulness(answer, _CONTEXT, mode="standard")
        assert result.faithful is False

    # ---- Negative: OUTSIDE knowledge must be rejected even if plausibly true ----
    async def test_outside_knowledge_is_rejected(self):
        answer = "NovaTech is headquartered in Austin, Texas."  # not in the context at all
        result = await evaluate_faithfulness(answer, _CONTEXT, mode="standard")
        assert result.faithful is False

    # ---- Negative: a projection stated as FACT must be rejected even in grounded mode ----
    async def test_projection_as_fact_is_rejected_grounded(self):
        # Grounded mode allows projections *framed as estimates*, but not invented
        # numbers presented as recorded fact.
        answer = "Q1 2025 revenue was $1.55B."
        result = await evaluate_faithfulness(answer, _CONTEXT, mode="grounded")
        assert result.faithful is False
