"""
Phase 4 tests — intent clarifier and intent eval.

Unit tests mock the LLM.
Integration tests call claude-sonnet-5 against the 5 assignment queries.
  Requires: ANTHROPIC_API_KEY + TEST_DATABASE_URL + NOVATECH_PDF indexed.
"""
import os
from unittest.mock import AsyncMock, patch

import pytest

NOVATECH_PDF = os.getenv("NOVATECH_PDF", "")


# ---------------------------------------------------------------------------
# Unit tests — intent eval logic (no LLM, no DB)
# ---------------------------------------------------------------------------

class TestIntentEval:
    def _make_intent(self, **kwargs):
        from pipeline.intent_clarifier import QueryIntent
        defaults = dict(
            query_type="textual",
            clarified_query="What were the Q4 highlights?",
            target_table=None,
            target_column=None,
            row_filter=None,
            aggregation=None,
            confidence=0.9,
            reasoning="Clear textual question.",
        )
        defaults.update(kwargs)
        return QueryIntent(**defaults)

    def test_high_confidence_textual_passes(self):
        from evals.intent_eval import evaluate_intent
        result = evaluate_intent(self._make_intent())
        assert result.passed is True
        assert result.reason is None

    def test_exactly_at_threshold_fails(self):
        from evals.intent_eval import evaluate_intent
        result = evaluate_intent(self._make_intent(confidence=0.74))
        assert result.passed is False

    def test_exactly_at_threshold_passes(self):
        from evals.intent_eval import evaluate_intent
        result = evaluate_intent(self._make_intent(confidence=0.75))
        assert result.passed is True

    def test_analytical_without_table_fails(self):
        from evals.intent_eval import evaluate_intent
        result = evaluate_intent(self._make_intent(
            query_type="analytical",
            target_table=None,
            aggregation="sum all values",
            confidence=0.9,
        ))
        assert result.passed is False
        assert "table" in result.reason.lower()

    def test_analytical_with_table_passes(self):
        from evals.intent_eval import evaluate_intent
        result = evaluate_intent(self._make_intent(
            query_type="analytical",
            target_table="Departmental Headcount",
            target_column="Headcount Q3",
            aggregation="sum all values",
            confidence=0.9,
        ))
        assert result.passed is True

    def test_hybrid_without_table_fails(self):
        from evals.intent_eval import evaluate_intent
        result = evaluate_intent(self._make_intent(
            query_type="hybrid",
            target_table=None,
            aggregation="compare trends",
            confidence=0.85,
        ))
        assert result.passed is False

    def test_hybrid_with_table_passes(self):
        from evals.intent_eval import evaluate_intent
        result = evaluate_intent(self._make_intent(
            query_type="hybrid",
            target_table="Departmental Headcount",
            aggregation="compare Q3 vs Q4 across all rows",
            confidence=0.85,
        ))
        assert result.passed is True


class TestSystemPromptBuilding:
    """_STATIC_INSTRUCTIONS (fully fixed) and _build_corpus_context (per-corpus)
    are kept as two separate prompt-cache breakpoints — see clarify_intent."""

    def test_static_instructions_contain_no_dynamic_content(self):
        from pipeline.intent_clarifier import _STATIC_INSTRUCTIONS
        # Must never vary by corpus, or the cache breakpoint is pointless.
        assert "Revenue Table" not in _STATIC_INSTRUCTIONS
        assert "Available schema" not in _STATIC_INSTRUCTIONS

    def test_corpus_context_contains_table_names(self):
        from pipeline.intent_clarifier import _build_corpus_context
        catalog = {"Revenue Table": ["Q3 Revenue", "Q4 Revenue"], "Headcount": ["Dept", "Count"]}
        prompt = _build_corpus_context(catalog, [])
        assert "Revenue Table" in prompt
        assert "Headcount" in prompt

    def test_corpus_context_contains_column_names(self):
        from pipeline.intent_clarifier import _build_corpus_context
        catalog = {"Revenue": ["Q3 Revenue", "Q4 Revenue"]}
        prompt = _build_corpus_context(catalog, [])
        assert "Q3 Revenue" in prompt
        assert "Q4 Revenue" in prompt

    def test_empty_catalog_handled_gracefully(self):
        from pipeline.intent_clarifier import _build_corpus_context
        prompt = _build_corpus_context({}, [])
        assert "no tables indexed" in prompt

    def test_corpus_context_contains_narrative_preview(self):
        from pipeline.intent_clarifier import _build_corpus_context
        prompt = _build_corpus_context({}, ["A new office will open in Austin"])
        assert "A new office will open in Austin" in prompt

    def test_empty_narrative_preview_handled_gracefully(self):
        from pipeline.intent_clarifier import _build_corpus_context
        prompt = _build_corpus_context({}, [])
        assert "no narrative text indexed" in prompt


class TestClarifyIntentMocked:
    async def test_returns_query_intent(self):
        from pipeline.intent_clarifier import QueryIntent, clarify_intent

        fake_intent = QueryIntent(
            query_type="textual",
            clarified_query="What were the main Q4 highlights?",
            confidence=0.95,
            reasoning="Narrative question.",
        )

        mock_session = AsyncMock()
        with patch("pipeline.intent_clarifier.get_schema_catalog", new_callable=AsyncMock) as mock_catalog, \
             patch("pipeline.intent_clarifier.get_narrative_preview", new_callable=AsyncMock) as mock_preview, \
             patch("pipeline.intent_clarifier.ChatAnthropic") as mock_llm_cls:

            mock_catalog.return_value = {}
            mock_preview.return_value = []
            mock_structured = AsyncMock(return_value=fake_intent)
            mock_llm_cls.return_value.with_structured_output.return_value.ainvoke = mock_structured

            result = await clarify_intent("What were the main Q4 highlights?", mock_session)

        assert isinstance(result, QueryIntent)
        assert result.query_type == "textual"
        assert result.confidence == 0.95

    async def test_static_instructions_precede_corpus_context(self):
        """Static instructions come before the per-corpus schema/narrative in
        the system prompt — keeps the stable part as a clean prefix."""
        from pipeline.intent_clarifier import QueryIntent, clarify_intent

        fake_intent = QueryIntent(
            query_type="textual", clarified_query="q",
            confidence=0.9, reasoning="r",
        )
        mock_session = AsyncMock()
        captured_messages = []

        async def capture_invoke(messages):
            captured_messages.extend(messages)
            return fake_intent

        with patch("pipeline.intent_clarifier.get_schema_catalog", new_callable=AsyncMock) as mock_catalog, \
             patch("pipeline.intent_clarifier.get_narrative_preview", new_callable=AsyncMock) as mock_preview, \
             patch("pipeline.intent_clarifier.ChatAnthropic") as mock_llm_cls:

            mock_catalog.return_value = {}
            mock_preview.return_value = []
            mock_llm_cls.return_value.with_structured_output.return_value.ainvoke = capture_invoke

            await clarify_intent("q", mock_session)

        system_content = captured_messages[0].content
        assert system_content.index("query classifier") < system_content.index("Available schema")


# ---------------------------------------------------------------------------
# Integration tests — real LLM calls against the 5 assignment queries
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAssignmentQueries:
    """
    The 5 example queries from the assignment brief.
    These hit the real claude-sonnet-5 model and require ANTHROPIC_API_KEY.
    """

    def setup_method(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        if not os.getenv("TEST_DATABASE_URL"):
            pytest.skip("TEST_DATABASE_URL not set")

    async def _clarify(self, question: str):
        from database.connection import get_async_session
        from pipeline.intent_clarifier import clarify_intent
        async with get_async_session() as session:
            return await clarify_intent(question, session)

    async def test_q4_highlights_is_textual(self):
        intent = await self._clarify("What were the main Q4 highlights?")
        assert intent.query_type == "textual"
        assert intent.confidence >= 0.75

    async def test_total_headcount_q3_is_analytical(self):
        intent = await self._clarify(
            "What was the total headcount across all departments in Q3?"
        )
        assert intent.query_type == "analytical"
        assert intent.confidence >= 0.75
        assert intent.target_table is not None
        assert intent.target_column is not None
        assert intent.aggregation is not None

    async def test_highest_headcount_dept_q4_is_analytical(self):
        intent = await self._clarify(
            "Which department had the highest headcount in Q4?"
        )
        assert intent.query_type == "analytical"
        assert intent.confidence >= 0.75
        assert intent.target_table is not None

    async def test_q3_vs_q4_comparison_is_analytical_or_hybrid(self):
        intent = await self._clarify(
            "How did Q3 headcount compare to Q4 across departments?"
        )
        assert intent.query_type in ("analytical", "hybrid")
        assert intent.confidence >= 0.75
        assert intent.target_table is not None

    async def test_hiring_prediction_is_hybrid(self):
        intent = await self._clarify(
            "Based on the headcount trends, which department is likely to need "
            "the most hiring in Q1?"
        )
        assert intent.query_type == "hybrid"
        assert intent.confidence >= 0.75


@pytest.mark.integration
class TestRetrievalStyleClassification:
    """retrieval_style must distinguish broad/summary questions (no single
    fact target — gated on cosine similarity downstream) from specific-fact
    questions (gated on the cross-encoder rerank_score). Hits the real
    claude-sonnet-5 model."""

    def setup_method(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        if not os.getenv("TEST_DATABASE_URL"):
            pytest.skip("TEST_DATABASE_URL not set")

    async def _clarify(self, question: str):
        from database.connection import get_async_session
        from pipeline.intent_clarifier import clarify_intent
        async with get_async_session() as session:
            return await clarify_intent(question, session)

    async def test_highlights_question_is_broad(self):
        intent = await self._clarify("What were the main Q4 highlights?")
        assert intent.retrieval_style == "broad"

    async def test_summarize_quarter_is_broad(self):
        intent = await self._clarify("Summarize the quarter.")
        assert intent.retrieval_style == "broad"

    async def test_company_performance_is_broad(self):
        intent = await self._clarify("How did the company perform in Q4?")
        assert intent.retrieval_style == "broad"

    async def test_specific_percentage_question_is_specific(self):
        intent = await self._clarify(
            "What percentage of total revenue came from cloud services?"
        )
        assert intent.retrieval_style == "specific"

    async def test_specific_location_question_is_specific(self):
        intent = await self._clarify("Where will the new R&D center be located?")
        assert intent.retrieval_style == "specific"
