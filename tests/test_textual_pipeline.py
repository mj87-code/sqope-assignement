"""
Phase 5 tests — retrieval eval, faithfulness eval, synthesizer, textual pipeline.

Unit tests mock LLM and tools.
Integration tests require ANTHROPIC_API_KEY + TEST_DATABASE_URL + indexed NovaTech PDF.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

NOVATECH_PDF = os.getenv("NOVATECH_PDF", "")


# ---------------------------------------------------------------------------
# Retrieval eval unit tests
# ---------------------------------------------------------------------------

class TestRetrievalEval:
    """evaluate_retrieval's gate depends on retrieval_style (set by the intent
    clarifier): 'specific' gates on rerank_score (sharp fact-matching
    separation), 'broad' gates on cosine similarity (rerank_score alone
    scores summary/overview queries near zero even against relevant content,
    since cross-encoders are trained for fact-matching, not topic-summary
    relevance)."""

    def test_empty_results_fails(self):
        from evals.retrieval_eval import evaluate_retrieval
        result = evaluate_retrieval([])
        assert result.passed is False
        assert result.top_score is None

    def test_specific_style_gates_on_rerank_score(self):
        from evals.retrieval_eval import RERANK_THRESHOLD, evaluate_retrieval
        low = RERANK_THRESHOLD - 0.1
        chunks = [{"rerank_score": low, "similarity": 0.99, "content": "x"}]
        result = evaluate_retrieval(chunks, "specific")
        assert result.passed is False
        assert result.top_score == pytest.approx(low)

    def test_specific_style_ignores_similarity(self):
        from evals.retrieval_eval import RERANK_THRESHOLD, evaluate_retrieval
        chunks = [{"rerank_score": RERANK_THRESHOLD, "similarity": 0.0, "content": "x"}]
        result = evaluate_retrieval(chunks, "specific")
        assert result.passed is True

    def test_broad_style_gates_on_similarity(self):
        from evals.retrieval_eval import SIMILARITY_THRESHOLD, evaluate_retrieval
        low = SIMILARITY_THRESHOLD - 0.1
        chunks = [{"rerank_score": 0.99, "similarity": low, "content": "x"}]
        result = evaluate_retrieval(chunks, "broad")
        assert result.passed is False
        assert result.top_score == pytest.approx(low)

    def test_broad_style_ignores_rerank_score(self):
        from evals.retrieval_eval import SIMILARITY_THRESHOLD, evaluate_retrieval
        chunks = [{"rerank_score": 0.0, "similarity": SIMILARITY_THRESHOLD, "content": "x"}]
        result = evaluate_retrieval(chunks, "broad")
        assert result.passed is True

    def test_default_style_is_specific(self):
        from evals.retrieval_eval import RERANK_THRESHOLD, evaluate_retrieval
        chunks = [{"rerank_score": RERANK_THRESHOLD, "similarity": 0.0, "content": "x"}]
        result = evaluate_retrieval(chunks)
        assert result.passed is True

    def test_gates_on_max_across_chunks(self):
        from evals.retrieval_eval import RERANK_THRESHOLD, evaluate_retrieval
        # The gate passes if ANY chunk clears the threshold, regardless of order.
        chunks = [
            {"rerank_score": RERANK_THRESHOLD - 0.1, "similarity": 0.0, "content": "below-threshold"},
            {"rerank_score": 0.95, "similarity": 0.0, "content": "above-threshold"},
        ]
        result = evaluate_retrieval(chunks, "specific")
        assert result.passed is True
        assert result.top_score == pytest.approx(0.95)

    def test_missing_rerank_score_raises_for_specific(self):
        from evals.retrieval_eval import evaluate_retrieval
        chunks = [{"similarity": 0.9, "content": "x"}]
        with pytest.raises(KeyError):
            evaluate_retrieval(chunks, "specific")

    def test_missing_similarity_raises_for_broad(self):
        from evals.retrieval_eval import evaluate_retrieval
        chunks = [{"rerank_score": 0.9, "content": "x"}]
        with pytest.raises(KeyError):
            evaluate_retrieval(chunks, "broad")


# ---------------------------------------------------------------------------
# Synthesizer unit tests (mocked LLM + faithfulness eval)
# ---------------------------------------------------------------------------

class TestSynthesizer:
    def _make_sources(self):
        from pipeline.synthesizer import SourceRef
        return [SourceRef(doc_filename="report.pdf", page_number=3, content_snippet="Revenue...")]

    async def test_returns_answer_when_faithful(self):
        from evals.faithfulness_eval import FaithfulnessResult
        from pipeline.synthesizer import synthesize

        with patch("pipeline.synthesizer.ChatAnthropic") as mock_llm, \
             patch("pipeline.synthesizer.evaluate_faithfulness", new_callable=AsyncMock) as mock_faith:

            mock_response = MagicMock()
            mock_response.answer = "Q4 revenue was $1.42B, up 14% YoY."
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            mock_faith.return_value = FaithfulnessResult(faithful=True, unsupported_claims=[])

            result = await synthesize(
                "What was Q4 revenue?",
                "Revenue was $1.42B in Q4 2024, up 14% year-over-year.",
                self._make_sources(),
                mode="standard",
            )

        assert result.eval_passed is True
        assert result.answer_basis == "indexed_documents"
        assert result.answer is not None

    async def test_insufficient_data_sentinel(self):
        from pipeline.synthesizer import synthesize

        with patch("pipeline.synthesizer.ChatAnthropic") as mock_llm, \
             patch("pipeline.synthesizer.evaluate_faithfulness", new_callable=AsyncMock) as mock_faith:

            mock_response = MagicMock()
            mock_response.answer = "INSUFFICIENT_DATA"
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )

            result = await synthesize(
                "What is the CEO salary?",
                "This document covers Q4 financials.",
                self._make_sources(),
            )

        assert result.eval_passed is False
        assert result.answer_basis == "insufficient_data"
        assert result.answer is None
        mock_faith.assert_not_called()

    async def test_faithfulness_failure_rejects_answer(self):
        from evals.faithfulness_eval import FaithfulnessResult
        from pipeline.synthesizer import synthesize

        with patch("pipeline.synthesizer.ChatAnthropic") as mock_llm, \
             patch("pipeline.synthesizer.evaluate_faithfulness", new_callable=AsyncMock) as mock_faith:

            mock_response = MagicMock()
            mock_response.answer = "Revenue was $2B and profit margins doubled."
            mock_llm.return_value.with_structured_output.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            mock_faith.return_value = FaithfulnessResult(
                faithful=False,
                unsupported_claims=["profit margins doubled"],
            )

            result = await synthesize(
                "Describe Q4.",
                "Revenue was $1.42B in Q4.",
                self._make_sources(),
            )

        assert result.eval_passed is False
        assert result.answer_basis == "eval_rejected"
        assert result.answer is None
        assert "profit margins doubled" in result.faithfulness_details["unsupported_claims"]

    async def test_grounded_mode_uses_grounded_prompt(self):
        from evals.faithfulness_eval import FaithfulnessResult
        from pipeline.synthesizer import _GROUNDED_SYSTEM, synthesize

        captured = []

        async def capture_invoke(messages):
            captured.extend(messages)
            m = MagicMock()
            m.answer = "R&D will likely need the most hiring."
            return m

        with patch("pipeline.synthesizer.ChatAnthropic") as mock_llm, \
             patch("pipeline.synthesizer.evaluate_faithfulness", new_callable=AsyncMock) as mock_faith:

            mock_llm.return_value.with_structured_output.return_value.ainvoke = capture_invoke
            mock_faith.return_value = FaithfulnessResult(faithful=True, unsupported_claims=[])

            await synthesize("prediction", "some context", self._make_sources(), mode="grounded")

        assert captured[0].content == _GROUNDED_SYSTEM


# ---------------------------------------------------------------------------
# Textual pipeline unit tests (mocked tools + synthesizer)
# ---------------------------------------------------------------------------

class TestTextualPipeline:
    def _make_intent(self, clarified_query="What were Q4 highlights?"):
        from pipeline.intent_clarifier import QueryIntent
        return QueryIntent(
            query_type="textual",
            clarified_query=clarified_query,
            confidence=0.95,
            reasoning="Narrative question.",
        )

    async def test_low_relevance_returns_insufficient_data(self):
        from pipeline.textual import run_textual

        with patch("pipeline.textual.search_text_chunks") as mock_search:
            mock_search.ainvoke = AsyncMock(return_value=[
                {"rerank_score": 0.4, "content": "x", "doc_filename": "r.pdf", "page_number": 1}
            ])
            result = await run_textual(self._make_intent())

        assert result.eval_passed is False
        assert result.answer_basis == "insufficient_data"

    async def test_no_results_returns_insufficient_data(self):
        from pipeline.textual import run_textual

        with patch("pipeline.textual.search_text_chunks") as mock_search:
            mock_search.ainvoke = AsyncMock(return_value=[])
            result = await run_textual(self._make_intent())

        assert result.eval_passed is False
        assert result.answer_basis == "insufficient_data"

    async def test_good_chunks_call_synthesizer_in_standard_mode(self):
        from pipeline.synthesizer import SourceRef, SynthesisResult
        from pipeline.textual import run_textual

        good_chunks = [
            {"rerank_score": 0.88, "content": "Revenue $1.42B Q4.", "doc_filename": "r.pdf", "page_number": 5},
        ]
        fake_result = SynthesisResult(
            answer="Q4 revenue was $1.42B.",
            sources=[SourceRef(doc_filename="r.pdf", page_number=5, content_snippet="Revenue...")],
            eval_passed=True,
            answer_basis="indexed_documents",
            rejection_reason=None,
        )

        with patch("pipeline.textual.search_text_chunks") as mock_search, \
             patch("pipeline.textual.synthesize", new_callable=AsyncMock) as mock_synth:

            mock_search.ainvoke = AsyncMock(return_value=good_chunks)
            mock_synth.return_value = fake_result

            result = await run_textual(self._make_intent())

        assert result.eval_passed is True
        call_args = mock_synth.call_args
        mode = call_args.kwargs.get("mode") or call_args.args[3]
        assert mode == "standard"

    async def test_sources_built_from_chunks(self):
        from pipeline.synthesizer import SynthesisResult
        from pipeline.textual import run_textual

        chunks = [
            {"rerank_score": 0.9, "content": "Revenue grew 14% YoY.", "doc_filename": "novatech.pdf", "page_number": 3},
            {"rerank_score": 0.8, "content": "Cloud revenue up 48%.", "doc_filename": "novatech.pdf", "page_number": 4},
        ]
        captured_sources = []

        async def capture_synth(question, context, sources, mode="standard"):
            captured_sources.extend(sources)
            return SynthesisResult(
                answer="Revenue grew.", sources=sources, eval_passed=True,
                answer_basis="indexed_documents", rejection_reason=None,
            )

        with patch("pipeline.textual.search_text_chunks") as mock_search, \
             patch("pipeline.textual.synthesize", new_callable=AsyncMock, side_effect=capture_synth):

            mock_search.ainvoke = AsyncMock(return_value=chunks)
            await run_textual(self._make_intent())

        assert len(captured_sources) == 2
        assert captured_sources[0].doc_filename == "novatech.pdf"
        assert captured_sources[0].page_number == 3


# ---------------------------------------------------------------------------
# Integration tests — real LLM + real DB with indexed NovaTech PDF
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestTextualIntegration:
    def setup_method(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        if not NOVATECH_PDF:
            pytest.skip("NOVATECH_PDF not set")

    def _make_intent(self, query: str):
        from pipeline.intent_clarifier import QueryIntent
        return QueryIntent(
            query_type="textual",
            clarified_query=query,
            confidence=0.9,
            reasoning="test",
        )

    async def test_q4_highlights_returns_answer(self):
        from pipeline.textual import run_textual
        result = await run_textual(self._make_intent("What were the main Q4 highlights?"))
        assert result.eval_passed is True
        assert result.answer is not None
        assert result.answer_basis == "indexed_documents"
        assert len(result.sources) > 0

    async def test_q4_highlights_mentions_revenue(self):
        from pipeline.textual import run_textual
        result = await run_textual(self._make_intent("What were the Q4 financial highlights?"))
        if result.eval_passed:
            assert any(
                term in result.answer.lower()
                for term in ["revenue", "1.42", "$1.4", "billion"]
            ), f"Expected revenue mention in: {result.answer}"

    async def test_unknown_query_returns_insufficient_data(self):
        from pipeline.textual import run_textual
        result = await run_textual(
            self._make_intent("What is the personal net worth of the CEO in 2019?")
        )
        assert result.answer_basis in ("insufficient_data", "eval_rejected")
