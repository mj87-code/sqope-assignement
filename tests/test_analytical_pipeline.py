"""
Analytical pipeline tests — text-to-SQL path.

Unit tests: SQL cleaning/validation, context formatting, pipeline routing
(LLM + DB mocked). Integration tests: real LLM writes SQL, real DB executes it.
"""
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

NOVATECH_PDF = os.getenv("NOVATECH_PDF", "")


@asynccontextmanager
async def _dummy_session():
    yield "SESSION"


def _make_intent(**kwargs):
    from pipeline.intent_clarifier import QueryIntent
    defaults = dict(
        query_type="analytical",
        clarified_query="What is the total Q3 headcount?",
        entities=["Q3", "headcount"],
        target_table="Table 3",
        target_column=None,
        row_filter=None,
        aggregation=None,
        confidence=0.92,
        reasoning="analytical sum question",
    )
    defaults.update(kwargs)
    return QueryIntent(**defaults)


# ---------------------------------------------------------------------------
# SQL cleaning + validation (pure)
# ---------------------------------------------------------------------------

class TestValidateSelect:
    def test_select_allowed(self):
        from pipeline.analytical import _validate_select
        ok, _ = _validate_select("SELECT * FROM table_rows")
        assert ok

    def test_with_cte_allowed(self):
        from pipeline.analytical import _validate_select
        ok, _ = _validate_select("WITH x AS (SELECT 1) SELECT * FROM x")
        assert ok

    def test_delete_rejected(self):
        from pipeline.analytical import _validate_select
        ok, reason = _validate_select("DELETE FROM table_rows")
        assert not ok and reason

    def test_update_rejected(self):
        from pipeline.analytical import _validate_select
        ok, _ = _validate_select("UPDATE table_rows SET x = 1")
        assert not ok

    def test_multiple_statements_rejected(self):
        from pipeline.analytical import _validate_select
        ok, reason = _validate_select("SELECT 1; DROP TABLE table_rows")
        assert not ok and "single" in reason


# ---------------------------------------------------------------------------
# _generate_sql prompt structure
# ---------------------------------------------------------------------------

class TestGenerateSqlPromptStructure:
    """Static instructions precede the per-corpus schema in the system
    prompt; target_table (per-query) lives in the HumanMessage instead."""

    async def test_static_instructions_precede_corpus_schema(self):
        from pipeline.analytical import _generate_sql

        captured_messages = []

        async def capture_invoke(messages):
            captured_messages.extend(messages)
            from pipeline.analytical import GeneratedSQL
            return GeneratedSQL(sql="SELECT 1", explanation="x")

        with patch("pipeline.analytical.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.with_structured_output.return_value.ainvoke = capture_invoke
            await _generate_sql(
                "What is the total headcount?",
                {"Headcount": ["Dept", "Q3"]},
                target_table="Headcount",
            )

        system_content = captured_messages[0].content
        assert system_content.index("translate a question") < system_content.index("Headcount")

    async def test_target_table_hint_in_human_message_not_system(self):
        from pipeline.analytical import _generate_sql

        captured_messages = []

        async def capture_invoke(messages):
            captured_messages.extend(messages)
            from pipeline.analytical import GeneratedSQL
            return GeneratedSQL(sql="SELECT 1", explanation="x")

        with patch("pipeline.analytical.ChatAnthropic") as mock_llm_cls:
            mock_llm_cls.return_value.with_structured_output.return_value.ainvoke = capture_invoke
            await _generate_sql(
                "What is the total headcount?",
                {"Headcount": ["Dept", "Q3"]},
                target_table="Headcount",
            )

        assert "most relevant table" not in captured_messages[0].content
        assert "most relevant table" in captured_messages[1].content


# ---------------------------------------------------------------------------
# _format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_includes_sql_and_result(self):
        from pipeline.analytical import _format_context
        ctx = _format_context(
            "SELECT SUM(...) AS total FROM table_rows",
            "Sum of Q3 headcount",
            [{"total": 8200}],
            "query_audited",
        )
        assert "QUERY RESULT" in ctx
        assert "SELECT SUM" in ctx
        assert "8200" in ctx

    def test_confirmed_tier_has_crosschecked_header(self):
        from pipeline.analytical import _format_context
        ctx = _format_context("SELECT 1", "x", [{"total": 1}], "independently_confirmed")
        assert "CROSS-CHECKED QUERY RESULT" in ctx

    def test_rows_capped_at_50(self):
        from pipeline.analytical import _format_context
        ctx = _format_context("SELECT 1", "x", [{"v": i} for i in range(100)], "query_audited")
        assert ctx.count("'v'") <= 50


# ---------------------------------------------------------------------------
# run_analytical routing (LLM + DB mocked)
# ---------------------------------------------------------------------------

class TestRunAnalytical:
    def _patches(self, catalog, generated=None, rows=None, raises=None,
                 sql_correct=True, cross=("not_applicable", None)):
        from evals.sql_eval import SqlEvalResult
        from pipeline.analytical import GeneratedSQL
        gen = generated or GeneratedSQL(
            sql="SELECT SUM((row_data->>'Headcount Q3')::numeric) AS total "
                "FROM table_rows WHERE table_name = 'Table 3'",
            explanation="Sum of Q3 headcount",
        )
        run_select = AsyncMock(side_effect=raises) if raises else AsyncMock(return_value=rows or [])
        # The SQL-correctness gate is an LLM call — mock it. Default verdict is
        # "correct"; pass sql_correct=False to exercise the wrong-query fallback.
        sql_eval = AsyncMock(return_value=SqlEvalResult(
            correct=sql_correct,
            issues=[] if sql_correct else ["targets the wrong column for this question"],
        ))
        # The deterministic cross-check hits the DB + pandas — mock its verdict.
        # Default "not_applicable" (no scalar cross-check); override per test.
        cross_check = AsyncMock(return_value=cross)
        return (
            patch("pipeline.analytical.get_async_session", _dummy_session),
            patch("pipeline.analytical.get_schema_catalog", AsyncMock(return_value=catalog)),
            patch("pipeline.analytical._generate_sql", AsyncMock(return_value=gen)),
            patch("pipeline.analytical.run_readonly_select", run_select),
            patch("pipeline.analytical.evaluate_sql", sql_eval),
            patch("pipeline.analytical._df_cross_check", cross_check),
        )

    def _textual_fallback_patch(self):
        """Patch the textual fallback with a recognisable sentinel result so a test
        can assert the analytical path delegated to it."""
        from pipeline.synthesizer import SynthesisResult
        sentinel = SynthesisResult(
            answer="From narrative text.", sources=[], eval_passed=True,
            answer_basis="indexed_documents", rejection_reason=None,
        )
        mock = AsyncMock(return_value=sentinel)
        return patch("pipeline.analytical.run_textual", mock), mock, sentinel

    async def test_empty_catalog_falls_back_to_textual(self):
        from pipeline.analytical import run_analytical
        p1, p2, p3, p4, p5, p6 = self._patches(catalog={})
        fb, mock, sentinel = self._textual_fallback_patch()
        with p1, p2, p3, p4, p5, p6, fb:
            result = await run_analytical(_make_intent())
        mock.assert_awaited_once()
        assert result is sentinel

    async def test_invalid_sql_falls_back_to_textual(self):
        from pipeline.analytical import GeneratedSQL, run_analytical
        bad = GeneratedSQL(sql="DELETE FROM table_rows", explanation="oops")
        p1, p2, p3, p4, p5, p6 = self._patches(catalog={"Table 3": ["Headcount Q3"]}, generated=bad)
        fb, mock, sentinel = self._textual_fallback_patch()
        with p1, p2, p3, p4, p5, p6, fb:
            result = await run_analytical(_make_intent())
        mock.assert_awaited_once()
        assert result is sentinel

    async def test_no_rows_falls_back_to_textual(self):
        from pipeline.analytical import run_analytical
        p1, p2, p3, p4, p5, p6 = self._patches(catalog={"Table 3": ["Headcount Q3"]}, rows=[])
        fb, mock, sentinel = self._textual_fallback_patch()
        with p1, p2, p3, p4, p5, p6, fb:
            result = await run_analytical(_make_intent())
        mock.assert_awaited_once()
        assert result is sentinel

    async def test_all_null_rows_falls_back_to_textual(self):
        # The reported failure: an over-constrained filter (or a figure that lives
        # in prose, not a table) returns rows whose values are all NULL.
        from pipeline.analytical import run_analytical
        p1, p2, p3, p4, p5, p6 = self._patches(
            catalog={"Table 3": ["Headcount Q3"]}, rows=[{"cash_reserve_q4": None}]
        )
        fb, mock, sentinel = self._textual_fallback_patch()
        with p1, p2, p3, p4, p5, p6, fb:
            result = await run_analytical(_make_intent())
        mock.assert_awaited_once()
        assert result is sentinel

    async def test_execution_error_falls_back_to_textual(self):
        from pipeline.analytical import run_analytical
        p1, p2, p3, p4, p5, p6 = self._patches(
            catalog={"Table 3": ["Headcount Q3"]}, raises=RuntimeError("bad sql")
        )
        fb, mock, sentinel = self._textual_fallback_patch()
        with p1, p2, p3, p4, p5, p6, fb:
            result = await run_analytical(_make_intent())
        mock.assert_awaited_once()
        assert result is sentinel

    async def test_insufficient_synthesis_falls_back_to_textual(self):
        # Rows have data, but the synthesizer can't answer from them → try text.
        from pipeline.analytical import run_analytical
        from pipeline.synthesizer import SynthesisResult

        async def insufficient_synth(question, context, sources, mode="standard"):
            return SynthesisResult(
                answer=None, sources=sources, eval_passed=False,
                answer_basis="insufficient_data", rejection_reason="not in tables",
            )

        p1, p2, p3, p4, p5, p6 = self._patches(
            catalog={"Table 3": ["Headcount Q3"]}, rows=[{"total": 8200}]
        )
        fb, mock, sentinel = self._textual_fallback_patch()
        with p1, p2, p3, p4, p5, p6, fb, \
             patch("pipeline.analytical.synthesize", new_callable=AsyncMock,
                   side_effect=insufficient_synth):
            result = await run_analytical(_make_intent())
        mock.assert_awaited_once()
        assert result is sentinel

    async def test_happy_path_uses_standard_mode_and_verified_context(self):
        from pipeline.analytical import run_analytical
        from pipeline.synthesizer import SynthesisResult

        captured = {}

        async def capture_synth(question, context, sources, mode="standard"):
            captured["context"] = context
            captured["mode"] = mode
            return SynthesisResult(
                answer="Total Q3 headcount is 8,200.", sources=sources,
                eval_passed=True, answer_basis="indexed_documents", rejection_reason=None,
            )

        p1, p2, p3, p4, p5, p6 = self._patches(
            catalog={"Table 3": ["Headcount Q3"]}, rows=[{"total": 8200}]
        )
        with p1, p2, p3, p4, p5, p6, \
             patch("pipeline.analytical.synthesize", new_callable=AsyncMock, side_effect=capture_synth):
            result = await run_analytical(_make_intent())

        assert result.eval_passed is True
        assert captured["mode"] == "standard"
        assert "QUERY RESULT" in captured["context"]
        assert "8200" in captured["context"]
        # Structured verified data is attached, independent of the prose.
        assert result.result["kind"] == "analytical"
        assert result.result["rows"] == [{"total": 8200}]
        assert "SELECT" in result.result["sql"]
        # Default cross-check verdict is not_applicable → LLM-audited-only tier.
        assert result.result["verification"] == "query_audited"

    async def test_cross_check_agreement_marks_independently_confirmed(self):
        # SQL result and the pandas recompute agree → the figure earns the stronger
        # trust tier and a cross-checked context label.
        from pipeline.analytical import run_analytical
        from pipeline.synthesizer import SynthesisResult

        captured = {}

        async def capture_synth(question, context, sources, mode="standard"):
            captured["context"] = context
            return SynthesisResult(
                answer="Total Q3 headcount is 8,200.", sources=sources,
                eval_passed=True, answer_basis="indexed_documents", rejection_reason=None,
            )

        agree = ("agree", {"sql_value": 8200.0, "df_value": 8200.0,
                           "operation": "sum", "column": "Headcount Q3",
                           "excluded_aggregate_rows": 0, "agreed": True})
        p1, p2, p3, p4, p5, p6 = self._patches(
            catalog={"Table 3": ["Headcount Q3"]}, rows=[{"total": 8200}], cross=agree
        )
        with p1, p2, p3, p4, p5, p6, \
             patch("pipeline.analytical.synthesize", new_callable=AsyncMock, side_effect=capture_synth):
            result = await run_analytical(_make_intent())

        assert result.eval_passed is True
        assert result.result["verification"] == "independently_confirmed"
        assert result.result["cross_check"]["agreed"] is True
        assert "CROSS-CHECKED QUERY RESULT" in captured["context"]

    async def test_cross_check_disagreement_discards_and_falls_back(self):
        # SQL says 9,999 but the independent recompute says 8,200 → hard red flag.
        # The number must be discarded before any LLM gate or synthesis runs.
        from pipeline.analytical import run_analytical

        disagree = ("disagree", {"sql_value": 9999.0, "df_value": 8200.0,
                                 "operation": "sum", "column": "Headcount Q3",
                                 "excluded_aggregate_rows": 0, "agreed": False})
        p1, p2, p3, p4, p5, p6 = self._patches(
            catalog={"Table 3": ["Headcount Q3"]}, rows=[{"total": 9999}], cross=disagree
        )
        fb, mock, sentinel = self._textual_fallback_patch()
        sql_gate = AsyncMock()
        synth = AsyncMock()
        with p1, p2, p3, p4, p5, p6, fb, \
             patch("pipeline.analytical.evaluate_sql", sql_gate), \
             patch("pipeline.analytical.synthesize", synth):
            result = await run_analytical(_make_intent())

        # Deterministic gate short-circuits: neither the LLM gate nor synthesis run.
        sql_gate.assert_not_awaited()
        synth.assert_not_awaited()
        mock.assert_awaited_once()
        assert result is sentinel

    async def test_wrong_sql_discarded_and_falls_back_to_textual(self):
        # The query runs cleanly and returns a number, but the SQL-correctness eval
        # judges it answers the WRONG question (wrong column/aggregate/filter). The
        # figure must be discarded — never synthesized into a "verified" answer —
        # and the path falls back to textual retrieval.
        from pipeline.analytical import run_analytical

        p1, p2, p3, p4, p5, p6 = self._patches(
            catalog={"Table 3": ["Headcount Q3"]},
            rows=[{"total": 999}],
            sql_correct=False,
        )
        fb, mock, sentinel = self._textual_fallback_patch()
        # synthesize must NOT be reached when the gate fails.
        synth = AsyncMock()
        with p1, p2, p3, p4, p5, p6, fb, \
             patch("pipeline.analytical.synthesize", synth):
            result = await run_analytical(_make_intent())

        synth.assert_not_awaited()
        mock.assert_awaited_once()
        assert result is sentinel


# ---------------------------------------------------------------------------
# Integration — real LLM writes SQL, real DB executes it
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAnalyticalIntegration:
    def setup_method(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        if not os.getenv("TEST_DATABASE_URL") and not os.getenv("DATABASE_URL"):
            pytest.skip("No database URL set")

    async def _headcount_table(self):
        from database.connection import get_async_session
        from database.table_store import get_schema_catalog
        async with get_async_session() as session:
            catalog = await get_schema_catalog(session)
        for tbl, cols in catalog.items():
            if any("headcount" in c.lower() for c in cols):
                return tbl
        return None

    async def test_q3_headcount_sum_equals_8200(self):
        from pipeline.analytical import run_analytical
        table = await self._headcount_table()
        if not table:
            pytest.skip("No headcount table — index NovaTech PDF first")

        intent = _make_intent(
            clarified_query="What is the total headcount across all departments in Q3?",
            target_table=table,
        )
        result = await run_analytical(intent)
        assert result.eval_passed is True
        assert "8200" in result.answer.replace(",", "")

    async def test_highest_headcount_dept_q4_is_rd(self):
        from pipeline.analytical import run_analytical
        table = await self._headcount_table()
        if not table:
            pytest.skip("No headcount table")

        intent = _make_intent(
            clarified_query="Which department had the highest headcount in Q4?",
            target_table=table,
        )
        result = await run_analytical(intent)
        if result.eval_passed:
            assert "r&d" in result.answer.lower() or "research" in result.answer.lower()
