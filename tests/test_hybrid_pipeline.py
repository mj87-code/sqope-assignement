"""
Phase 7 tests — hybrid pipeline.

Unit tests mock tools and synthesizer.
Integration tests require ANTHROPIC_API_KEY + TEST_DATABASE_URL + indexed NovaTech PDF.
"""
import os
from unittest.mock import AsyncMock, patch

import pytest

NOVATECH_PDF = os.getenv("NOVATECH_PDF", "")

SAMPLE_ROWS = [
    {"Department": "R&D",         "Headcount Q3": 2450.0, "Headcount Q4": 2620.0},
    {"Department": "Sales",       "Headcount Q3": 1850.0, "Headcount Q4": 1920.0},
    {"Department": "Engineering", "Headcount Q3": 2100.0, "Headcount Q4": 2200.0},
]

SAMPLE_CHUNKS = [
    {"rerank_score": 0.87, "content": "Q4 saw strong growth across all divisions.",
     "doc_filename": "novatech.pdf", "page_number": 2},
    {"rerank_score": 0.81, "content": "R&D headcount grew by 7% from Q3 to Q4.",
     "doc_filename": "novatech.pdf", "page_number": 4},
]


def _make_intent(**kwargs):
    from pipeline.intent_clarifier import QueryIntent
    defaults = dict(
        query_type="hybrid",
        clarified_query="Which department is likely to need the most hiring in Q1?",
        entities=["Q1", "hiring", "department"],
        target_table="Departmental Headcount",
        target_column="Headcount Q4",
        row_filter=None,
        aggregation="find the row with the highest value",
        confidence=0.88,
        reasoning="Prediction requires both headcount trends and narrative context.",
    )
    defaults.update(kwargs)
    return QueryIntent(**defaults)


# ---------------------------------------------------------------------------
# _format_hybrid_context unit tests
# ---------------------------------------------------------------------------

class TestFormatHybridContext:
    def test_includes_text_section_when_chunks_present(self):
        from pipeline.hybrid import _format_hybrid_context
        ctx = _format_hybrid_context(SAMPLE_CHUNKS, SAMPLE_ROWS, None, None)
        assert "=== Document Text ===" in ctx
        assert "Q4 saw strong growth" in ctx

    def test_includes_table_section_when_rows_present(self):
        from pipeline.hybrid import _format_hybrid_context
        ctx = _format_hybrid_context([], SAMPLE_ROWS, None, None)
        assert "=== Table Data ===" in ctx
        assert "R&D" in ctx

    def test_omits_text_section_when_no_chunks(self):
        from pipeline.hybrid import _format_hybrid_context
        ctx = _format_hybrid_context([], SAMPLE_ROWS, None, None)
        assert "=== Document Text ===" not in ctx

    def test_computed_result_injected(self):
        from pipeline.hybrid import _format_hybrid_context
        from pipeline.table_compute import _Computed
        computed = _Computed("max", 2620.0, {"Department": "R&D", "Headcount Q4": 2620.0})
        ctx = _format_hybrid_context(SAMPLE_CHUNKS, SAMPLE_ROWS, "Headcount Q4", computed)
        assert "VERIFIED COMPUTED RESULT" in ctx
        assert "2620.0" in ctx
        assert "R&D" in ctx

    def test_source_page_numbers_in_context(self):
        from pipeline.hybrid import _format_hybrid_context
        ctx = _format_hybrid_context(SAMPLE_CHUNKS, [], None, None)
        assert "page 2" in ctx
        assert "page 4" in ctx


class TestRelevantEnough:
    """_relevant_enough's gate depends on retrieval_style: 'specific' gates on
    rerank_score, 'broad' gates on cosine similarity — same rationale as
    evals/retrieval_eval (rerank_score alone scores summary/overview queries
    near zero even against relevant content)."""

    def test_specific_style_gates_on_rerank_score(self):
        from pipeline.hybrid import _HYBRID_RERANK_FLOOR, _relevant_enough
        above = {"rerank_score": _HYBRID_RERANK_FLOOR + 0.1, "similarity": 0.0}
        below = {"rerank_score": _HYBRID_RERANK_FLOOR - 0.1, "similarity": 1.0}
        assert _relevant_enough(above, "specific") is True
        assert _relevant_enough(below, "specific") is False

    def test_broad_style_gates_on_similarity(self):
        from pipeline.hybrid import _HYBRID_SIMILARITY_FLOOR, _relevant_enough
        above = {"rerank_score": 0.0, "similarity": _HYBRID_SIMILARITY_FLOOR + 0.1}
        below = {"rerank_score": 1.0, "similarity": _HYBRID_SIMILARITY_FLOOR - 0.1}
        assert _relevant_enough(above, "broad") is True
        assert _relevant_enough(below, "broad") is False

    def test_missing_rerank_score_raises_for_specific(self):
        from pipeline.hybrid import _relevant_enough
        # No silent fallback: a chunk missing rerank_score means the reranker
        # was skipped, which should never happen — tools/reranker.py raises
        # rather than returning chunks in that shape.
        with pytest.raises(KeyError):
            _relevant_enough({"similarity": 0.9}, "specific")

    def test_missing_similarity_raises_for_broad(self):
        from pipeline.hybrid import _relevant_enough
        with pytest.raises(KeyError):
            _relevant_enough({"rerank_score": 0.9}, "broad")


# ---------------------------------------------------------------------------
# run_hybrid pipeline unit tests
# ---------------------------------------------------------------------------

class TestHybridPipeline:
    async def test_both_insufficient_returns_insufficient_data(self):
        from pipeline.hybrid import run_hybrid

        with patch("pipeline.hybrid.search_text_chunks") as mock_search, \
             patch("pipeline.hybrid.get_table_rows") as mock_rows:
            mock_search.ainvoke = AsyncMock(return_value=[])
            mock_rows.ainvoke = AsyncMock(return_value=[])

            result = await run_hybrid(_make_intent())

        assert result.eval_passed is False
        assert result.answer_basis == "insufficient_data"

    async def test_table_only_proceeds_without_text(self):
        from pipeline.hybrid import run_hybrid
        from pipeline.synthesizer import SynthesisResult

        with patch("pipeline.hybrid.search_text_chunks") as mock_search, \
             patch("pipeline.hybrid.get_table_rows") as mock_rows, \
             patch("pipeline.hybrid.synthesize", new_callable=AsyncMock) as mock_synth:

            # low-relevance chunks (text retrieval fails), but table data available
            mock_search.ainvoke = AsyncMock(return_value=[
                {"rerank_score": 0.3, "content": "x", "doc_filename": "f.pdf", "page_number": 1}
            ])
            mock_rows.ainvoke = AsyncMock(return_value=SAMPLE_ROWS)
            mock_synth.return_value = SynthesisResult(
                answer="R&D.", sources=[], eval_passed=True,
                answer_basis="indexed_documents", rejection_reason=None,
            )

            result = await run_hybrid(_make_intent())

        assert result.eval_passed is True
        mock_synth.assert_called_once()

    async def test_uses_grounded_mode(self):
        from pipeline.hybrid import run_hybrid
        from pipeline.synthesizer import SynthesisResult

        with patch("pipeline.hybrid.search_text_chunks") as mock_search, \
             patch("pipeline.hybrid.get_table_rows") as mock_rows, \
             patch("pipeline.hybrid.synthesize", new_callable=AsyncMock) as mock_synth:

            mock_search.ainvoke = AsyncMock(return_value=SAMPLE_CHUNKS)
            mock_rows.ainvoke = AsyncMock(return_value=SAMPLE_ROWS)
            mock_synth.return_value = SynthesisResult(
                answer="R&D will need the most hiring.", sources=[], eval_passed=True,
                answer_basis="indexed_documents", rejection_reason=None,
            )

            await run_hybrid(_make_intent())

        call_args = mock_synth.call_args
        mode = call_args.kwargs.get("mode") or call_args.args[3]
        assert mode == "grounded"

    async def test_computed_result_in_context(self):
        from pipeline.hybrid import run_hybrid
        from pipeline.synthesizer import SynthesisResult

        captured_context = []

        async def capture_synth(question, context, sources, mode="standard"):
            captured_context.append(context)
            return SynthesisResult(
                answer="R&D.", sources=[], eval_passed=True,
                answer_basis="indexed_documents", rejection_reason=None,
            )

        with patch("pipeline.hybrid.search_text_chunks") as mock_search, \
             patch("pipeline.hybrid.get_table_rows") as mock_rows, \
             patch("pipeline.hybrid.synthesize", new_callable=AsyncMock, side_effect=capture_synth):

            mock_search.ainvoke = AsyncMock(return_value=SAMPLE_CHUNKS)
            mock_rows.ainvoke = AsyncMock(return_value=SAMPLE_ROWS)

            await run_hybrid(_make_intent(
                target_column="Headcount Q4",
                aggregation="find the row with the highest value",
            ))

        assert "VERIFIED COMPUTED RESULT" in captured_context[0]
        assert "2620.0" in captured_context[0]

    async def test_sources_come_from_text_chunks(self):
        from pipeline.hybrid import run_hybrid
        from pipeline.synthesizer import SynthesisResult

        captured_sources = []

        async def capture_synth(question, context, sources, mode="standard"):
            captured_sources.extend(sources)
            return SynthesisResult(
                answer="ok.", sources=sources, eval_passed=True,
                answer_basis="indexed_documents", rejection_reason=None,
            )

        with patch("pipeline.hybrid.search_text_chunks") as mock_search, \
             patch("pipeline.hybrid.get_table_rows") as mock_rows, \
             patch("pipeline.hybrid.synthesize", new_callable=AsyncMock, side_effect=capture_synth):

            mock_search.ainvoke = AsyncMock(return_value=SAMPLE_CHUNKS)
            mock_rows.ainvoke = AsyncMock(return_value=SAMPLE_ROWS)

            await run_hybrid(_make_intent())

        assert len(captured_sources) == 2
        assert captured_sources[0].doc_filename == "novatech.pdf"

    async def test_no_sources_when_text_retrieval_fails(self):
        from pipeline.hybrid import run_hybrid
        from pipeline.synthesizer import SynthesisResult

        captured_sources = []

        async def capture_synth(question, context, sources, mode="standard"):
            captured_sources.extend(sources)
            return SynthesisResult(
                answer="ok.", sources=sources, eval_passed=True,
                answer_basis="indexed_documents", rejection_reason=None,
            )

        with patch("pipeline.hybrid.search_text_chunks") as mock_search, \
             patch("pipeline.hybrid.get_table_rows") as mock_rows, \
             patch("pipeline.hybrid.synthesize", new_callable=AsyncMock, side_effect=capture_synth):

            mock_search.ainvoke = AsyncMock(return_value=[
                {"rerank_score": 0.2, "content": "irrelevant", "doc_filename": "f.pdf", "page_number": 1}
            ])
            mock_rows.ainvoke = AsyncMock(return_value=SAMPLE_ROWS)

            await run_hybrid(_make_intent())

        assert captured_sources == []


# ---------------------------------------------------------------------------
# Computed/prose consistency guard
# ---------------------------------------------------------------------------

# Mirrors the real NovaTech Income Statement: revenue line items mixed with
# non-revenue metrics that carry no "total"/"subtotal" label, so nothing trips
# _is_aggregate_row's exclusion for them.
INCOME_STATEMENT_ROWS = [
    {"Category": "Revenue - Cloud Services",       "Q4 2024 (USD M)": 680.0},
    {"Category": "Revenue - AI Solutions",          "Q4 2024 (USD M)": 310.0},
    {"Category": "Revenue - Hardware",              "Q4 2024 (USD M)": 250.0},
    {"Category": "Revenue - Support & Maintenance", "Q4 2024 (USD M)": 180.0},
    {"Category": "Total Revenue",                   "Q4 2024 (USD M)": 1420.0},
    {"Category": "Cost of Goods Sold",              "Q4 2024 (USD M)": 730.0},
    {"Category": "Gross Profit",                    "Q4 2024 (USD M)": 690.0},
    {"Category": "Operating Expenses",               "Q4 2024 (USD M)": 420.0},
    {"Category": "Operating Income",                "Q4 2024 (USD M)": 270.0},
    {"Category": "Net Income",                       "Q4 2024 (USD M)": 230.0},
]


class TestComputedAgreesWithAnswer:
    def test_match_with_thousands_separator(self):
        from pipeline.hybrid import _computed_agrees_with_answer
        assert _computed_agrees_with_answer(1420.0, "The total was $1,420M.") is True

    def test_match_across_unit_scaling(self):
        from pipeline.hybrid import _computed_agrees_with_answer
        # column in USD millions, prose restated in billions
        assert _computed_agrees_with_answer(1420.0, "roughly $1.42 billion") is True

    def test_contradiction_detected(self):
        from pipeline.hybrid import _computed_agrees_with_answer
        assert _computed_agrees_with_answer(3760.0, "Revenue totalled $1,420M.") is False

    def test_answer_without_numbers_counts_as_agreement(self):
        from pipeline.hybrid import _computed_agrees_with_answer
        assert _computed_agrees_with_answer(2620.0, "R&D grew the fastest.") is True


class TestComputedConsistencyGuard:
    """result["computed"] is documented as machine-checkable and independent of
    the prose — it must never ship a figure that contradicts the answer it
    accompanies. On a mixed table, _try_compute can silently aggregate the
    wrong row set (it only excludes literal total/subtotal labels) while the
    synthesizer, reading the raw rows in context, answers correctly — observed
    live: computed 3760.0 alongside a correct prose answer of $1,420M."""

    def _guard_intent(self):
        return _make_intent(
            clarified_query="What is the combined Q4 total from just the revenue line items?",
            target_table="Table 1: Consolidated Income Statement (USD Millions)",
            target_column="Q4 2024 (USD M)",
            aggregation="sum the revenue line items only",
        )

    async def _run_with_answer(self, answer: str):
        from pipeline.hybrid import run_hybrid
        from pipeline.synthesizer import SynthesisResult

        with patch("pipeline.hybrid.search_text_chunks") as mock_search, \
             patch("pipeline.hybrid.get_table_rows") as mock_rows, \
             patch("pipeline.hybrid.synthesize", new_callable=AsyncMock) as mock_synth:

            mock_search.ainvoke = AsyncMock(return_value=[])
            mock_rows.ainvoke = AsyncMock(return_value=INCOME_STATEMENT_ROWS)
            mock_synth.return_value = SynthesisResult(
                answer=answer, sources=[], eval_passed=True,
                answer_basis="indexed_documents", rejection_reason=None,
            )
            return await run_hybrid(self._guard_intent())

    async def test_contradicting_computed_is_dropped(self):
        # _try_compute sums every non-"total" row here (= 3760.0); the prose
        # correctly says 1,420 — the contradicting figure must not be attached.
        result = await self._run_with_answer(
            "The combined Q4 total for the four revenue line items is $1,420M."
        )
        assert result.answer is not None
        assert result.result["computed"] is None
        assert result.result["rows"]  # raw rows still attached for inspection

    async def test_agreeing_computed_is_kept(self):
        result = await self._run_with_answer(
            "The sum across all listed line items is $3,760M."
        )
        assert result.result["computed"] is not None
        assert result.result["computed"]["value"] == 3760.0

    async def test_numberless_answer_keeps_computed(self):
        result = await self._run_with_answer(
            "The revenue line items dominate the quarter."
        )
        assert result.result["computed"] is not None


# ---------------------------------------------------------------------------
# Integration tests — real LLM + DB with indexed NovaTech PDF
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestHybridIntegration:
    def setup_method(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        if not NOVATECH_PDF:
            pytest.skip("NOVATECH_PDF not set")

    async def _get_headcount_table_and_q4_col(self):
        from database.connection import get_async_session
        from database.table_store import get_all_table_names, get_schema_catalog
        async with get_async_session() as session:
            tables = await get_all_table_names(session)
            catalog = await get_schema_catalog(session)

        headcount_tables = [t for t in tables if "headcount" in t.lower() or "department" in t.lower()]
        if not headcount_tables:
            pytest.skip("No headcount table — index NovaTech PDF first")

        table = headcount_tables[0]
        q4_cols = [c for c in catalog.get(table, []) if "q4" in c.lower()]
        return table, q4_cols[0] if q4_cols else None

    async def test_hiring_prediction_returns_grounded_answer(self):
        from pipeline.hybrid import run_hybrid

        table, q4_col = await self._get_headcount_table_and_q4_col()

        intent = _make_intent(
            clarified_query="Based on the headcount trends, which department is likely to need the most hiring in Q1?",
            target_table=table,
            target_column=q4_col,
            aggregation="find the row with the highest value to identify growth trend",
        )
        result = await run_hybrid(intent)

        assert result.eval_passed is True
        assert result.answer is not None
        assert result.answer_basis == "indexed_documents"

    async def test_prediction_answer_references_rd(self):
        from pipeline.hybrid import run_hybrid

        table, q4_col = await self._get_headcount_table_and_q4_col()

        intent = _make_intent(
            clarified_query="Based on headcount trends from Q3 to Q4, which department will likely need the most new hires in Q1?",
            target_table=table,
            target_column=q4_col,
            aggregation="find the row with the highest value",
        )
        result = await run_hybrid(intent)

        if result.eval_passed:
            assert "r&d" in result.answer.lower() or "research" in result.answer.lower(), \
                f"Expected R&D in prediction: {result.answer}"
