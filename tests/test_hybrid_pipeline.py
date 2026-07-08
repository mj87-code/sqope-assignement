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


# Mirrors the real NovaTech "Consolidated Income Statement" table (pulled
# directly from the live indexed document) — four revenue line items, a
# "Total Revenue" subtotal (correctly excludable by _is_aggregate_row), and
# several NON-revenue rows that are themselves derived subtotals but aren't
# labelled "total" either, so nothing excludes them.
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


class TestHybridComputedResultContradiction:
    """KNOWN LIMITATION, currently failing on purpose.

    Reproduces a bug first found by manually querying the live system: for
    "what's the combined Q4 total from just the revenue line items", the
    verbose-mode call trace showed query_type=hybrid, hybrid.try_compute
    value=3760.0 (it summed every non-"total"-labelled row: the four revenue
    lines PLUS Cost of Goods Sold, Gross Profit, Operating Expenses, Operating
    Income, and Net Income) — while the synthesizer's own prose reasoning
    over the raw rows in context correctly said $1,420M.

    run_hybrid() attaches _try_compute's output to synthesis.result["computed"]
    unconditionally, with no check against what the synthesized answer actually
    says. The result: api/schemas.py::QueryResult.computed is documented as
    "machine-checkable verified data ... independent of the prose ... assert
    the exact figure here without parsing the answer text" — but here it
    silently CONTRADICTS a prose answer that happens to be correct. A reviewer
    who trusts the structured field over the prose (exactly what that field
    exists for) gets the wrong number.

    This test is expected to fail until run_hybrid either (a) suppresses an
    uncategorized/wrongly-scoped _try_compute figure instead of presenting it
    unconditionally, or (b) cross-checks it against the synthesized answer the
    way the analytical path cross-checks SQL against pandas. Do not "fix" it
    by loosening the assertion.
    """

    async def test_computed_result_does_not_contradict_synthesized_answer(self):
        from pipeline.hybrid import run_hybrid
        from pipeline.synthesizer import SynthesisResult

        # Mirrors the live incident: the LLM's prose answer correctly scopes to
        # just the four revenue line items and states $1,420M.
        async def capture_synth(question, context, sources, mode="standard"):
            return SynthesisResult(
                answer="Combined Q4 total for the four revenue line items = $1,420M.",
                sources=[], eval_passed=True,
                answer_basis="indexed_documents", rejection_reason=None,
            )

        with patch("pipeline.hybrid.search_text_chunks") as mock_search, \
             patch("pipeline.hybrid.get_table_rows") as mock_rows, \
             patch("pipeline.hybrid.synthesize", new_callable=AsyncMock, side_effect=capture_synth):

            mock_search.ainvoke = AsyncMock(return_value=[])
            mock_rows.ainvoke = AsyncMock(return_value=INCOME_STATEMENT_ROWS)

            result = await run_hybrid(_make_intent(
                target_table="Table 1: Consolidated Income Statement (USD Millions)",
                target_column="Q4 2024 (USD M)",
                aggregation="sum the revenue line items only",
            ))

        assert result.answer is not None
        assert "1,420" in result.answer or "1420" in result.answer

        # The machine-checkable figure must not contradict the prose it's
        # supposedly backing up. Today it does: computed.value is 3760 (every
        # non-"total"-labelled row summed), not 1420.
        assert result.result["computed"]["value"] == 1420.0


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
