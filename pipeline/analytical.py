"""
Analytical pipeline path — text-to-SQL over the indexed tables.

The LLM writes ONE read-only SELECT against the JSONB table store, grounded in the
live schema. PostgreSQL computes the result — the LLM never produces the number
itself. The query is validated as a single SELECT and executed inside a READ ONLY
transaction, then passes two verification gates before the figure is trusted:

  1. Deterministic cross-check — for single-column scalar aggregations, the same
     number is independently recomputed with pandas (`table_compute._try_compute`,
     the same calculator the hybrid path uses) and compared. Disagreement is a hard
     red flag → discard. This gate doesn't share the LLM's failure mode.
  2. SQL-correctness eval (LLM) — always-on backstop that audits the query against
     the *question* (right column/filter), catching intent-level errors gate 1
     can't see (both sides use the same intent).

The result carries a trust tier: `independently_confirmed` when the cross-check
agreed, else `query_audited` (LLM-audited only) — so the response never overstates
how strongly a figure was verified. A query that runs cleanly but fails a gate is
discarded in favour of textual retrieval rather than presented as verified.

This replaces the earlier keyword→pandas approach: SQL handles arbitrary
aggregations/filters/grouping, and the database (not a brittle keyword matcher)
decides the operation.
"""
import logging
import math
import re
from typing import cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from database.connection import get_async_session
from database.table_store import (
    format_rows,
    format_schema_catalog,
    get_schema_catalog,
    load_table_rows,
    run_readonly_select,
)
from evals.sql_eval import JUDGE_MODEL as SQL_EVAL_MODEL
from evals.sql_eval import evaluate_sql
from pipeline.intent_clarifier import QueryIntent
from pipeline.synthesizer import SourceRef, SynthesisResult, synthesize
from pipeline.table_compute import _try_compute
from pipeline.textual import run_textual

MODEL = "claude-sonnet-5"
MAX_SQL_ATTEMPTS = 2
# Cross-check float tolerance (Postgres numeric vs pandas float).
_CROSS_CHECK_TOL = 1e-6
log = logging.getLogger("pipeline.analytical")


class GeneratedSQL(BaseModel):
    sql: str = Field(
        description="A single read-only SELECT (or WITH ... SELECT) query answering the question."
    )
    explanation: str = Field(
        description="One sentence describing what the query computes."
    )


def _clean_sql(sql: str) -> str:
    """Strip markdown fences and a single trailing semicolon."""
    s = sql.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    return s.rstrip(";").strip()


def _validate_select(sql: str) -> tuple[bool, str | None]:
    """Allow exactly one SELECT/WITH statement. Real enforcement is the READ ONLY txn.
    Accepted trade-off: this doesn't block expensive-but-valid SELECTs (e.g. pg_sleep,
    a costly scan) — the 15s statement_timeout on the connection bounds the damage
    per query. No rate limiting exists on this API; revisit if it's ever exposed
    beyond trusted internal due-diligence users."""
    if not sql:
        return False, "empty query"
    if ";" in sql:
        return False, "only a single statement is allowed"
    head = sql.lstrip("(").lower()
    if not (head.startswith("select") or head.startswith("with")):
        return False, "only SELECT queries are permitted"
    return True, None


_STATIC_SQL_INSTRUCTIONS = """You translate a question into ONE read-only PostgreSQL SELECT query.

All indexed table data lives in a single table with only THREE real columns:

  table_rows(table_name TEXT, row_data JSONB, row_index INT)

Every value from a PDF table is a KEY inside the row_data JSONB object — it is NOT
a real column. You MUST read each value via row_data, everywhere it appears
(SELECT, WHERE, ORDER BY, GROUP BY):

- As text:   row_data->>'Column Name'
- As number: (row_data->>'Column Name')::numeric

CRITICAL: never write a bare quoted identifier like "Column Name" — that column
does not exist and the query will fail. The only bare identifiers allowed are
table_name, row_data, row_index.

Examples (substitute the actual table and column names from the schema below):

  -- aggregate a numeric column
  SELECT SUM((row_data->>'<numeric column>')::numeric) AS total
  FROM table_rows
  WHERE table_name = '<exact table name>';

  -- read several columns from specific rows
  SELECT row_data->>'<label column>'              AS label,
         (row_data->>'<numeric column>')::numeric AS value
  FROM table_rows
  WHERE table_name = '<exact table name>'
    AND row_data->>'<label column>' ILIKE '%<filter text>%';

Rules:
1. Return exactly ONE statement: a SELECT (or WITH ... SELECT). Never INSERT/UPDATE/DELETE/DDL.
2. Access every PDF value through row_data->>'...'; only table_name/row_data/row_index are real columns.
3. Cast to ::numeric before arithmetic or numeric ordering.
4. If a table mixes line items with a subtotal/total row (e.g. a row labelled
   'Total ...') and the user asks for a sum or maximum of the components, exclude
   those rows (e.g. AND row_data->>'<label column>' NOT ILIKE '%total%').
5. Alias computed columns with clear names (e.g. AS total_value)."""


def _build_sql_corpus_context(catalog: dict[str, list[str]]) -> str:
    return f"""Available tables and their row_data keys (use these names exactly):
{format_schema_catalog(catalog)}"""


async def _generate_sql(
    question: str,
    catalog: dict[str, list[str]],
    target_table: str | None,
    previous_sql: str | None = None,
    error: str | None = None,
) -> GeneratedSQL:
    # Suppressed: ChatAnthropic's other constructor fields use a
    # Field(None, alias=...) style basedpyright's pydantic support doesn't
    # recognise as having a default, so it misreports them as missing — a
    # false positive in the third-party stub, not a real missing-argument bug.
    llm = ChatAnthropic(model=MODEL)  # pyright: ignore[reportCallIssue]
    llm = llm.with_structured_output(GeneratedSQL, method="json_schema")

    # target_table is per-query, so it goes in the HumanMessage rather than
    # the system prompt.
    system_prompt = f"{_STATIC_SQL_INSTRUCTIONS}\n\n{_build_sql_corpus_context(catalog)}"
    hint = (
        f'\n\nThe most relevant table for this question is "{target_table}".'
        if target_table
        else ""
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{question}{hint}"),
    ]
    if previous_sql and error:
        messages.append(
            HumanMessage(
                content=(
                    f"Your previous query failed:\n{previous_sql}\n\n"
                    f"PostgreSQL error:\n{error}\n\n"
                    "Fix it. Remember: every PDF value is a JSONB key accessed via "
                    "row_data->>'Key' — only table_name, row_data, row_index are real columns."
                )
            )
        )
    # with_structured_output()'s return type is generically `dict | BaseModel`
    # regardless of the schema passed in — cast to the schema we know it is.
    return cast(GeneratedSQL, await llm.ainvoke(messages))


_CONFIRMED_HEADER = (
    "CROSS-CHECKED QUERY RESULT (computed by PostgreSQL AND independently recomputed "
    "by a second method — the two agree; values below are exact; do not round or modify):"
)
_AUDITED_HEADER = (
    "QUERY RESULT (computed by PostgreSQL, query logic audited; values below are exact "
    "but were NOT independently recomputed; do not round or modify):"
)


def _format_context(
    sql: str, explanation: str, rows: list[dict], verification: str
) -> str:
    header = _CONFIRMED_HEADER if verification == "independently_confirmed" else _AUDITED_HEADER
    lines = [
        header,
        f"  Purpose: {explanation}",
        f"  SQL: {sql}",
        "  Result rows:",
    ]
    lines.extend(format_rows(rows, indent="    "))
    return "\n".join(lines)


def _extract_sql_scalar(rows: list[dict]) -> float | None:
    """Return the single numeric value a scalar-aggregation query produced (one row,
    exactly one numeric cell — a label column alongside it is fine). None otherwise,
    meaning the result isn't a single comparable number."""
    if len(rows) != 1:
        return None
    nums = [
        v for v in rows[0].values()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    return float(nums[0]) if len(nums) == 1 else None


async def _df_cross_check(
    intent: QueryIntent, sql_rows: list[dict]
) -> tuple[str, dict | None]:
    """
    Independently recompute the SQL's scalar with pandas and compare — a
    deterministic second opinion that doesn't share the LLM's failure mode.

    Restricted to the clean case (single-column aggregation, no row filter): this
    keeps the check high-precision so a 'disagree' is a real red flag, not filter
    noise. Anything outside that returns 'not_applicable' and defers to the LLM
    SQL-correctness gate. Returns (verdict, detail) where verdict is
    'agree' | 'disagree' | 'not_applicable'.
    """
    # Clean case only — see the DD note: a mis-scoped filter would cause false
    # disagreements and erode trust in the gate.
    if not intent.target_column or intent.row_filter or not intent.target_table:
        return "not_applicable", None

    sql_scalar = _extract_sql_scalar(sql_rows)
    if sql_scalar is None:
        return "not_applicable", None

    async with get_async_session() as session:
        # limit=None: this cross-checks a full-table SQL aggregate against a pandas
        # recompute — a partial fetch would produce a false "disagree" for any
        # table over the default row cap and discard a correct SQL answer.
        raw_rows = await load_table_rows(session, intent.target_table, None, limit=None)
    computed = _try_compute(raw_rows, intent.target_column, intent.aggregation or "")
    if computed is None:
        return "not_applicable", None

    df_value = float(computed.value)
    agreed = math.isclose(sql_scalar, df_value, rel_tol=_CROSS_CHECK_TOL, abs_tol=_CROSS_CHECK_TOL)
    detail = {
        "sql_value": sql_scalar,
        "df_value": df_value,
        "operation": computed.operation,
        "column": intent.target_column,
        "excluded_aggregate_rows": computed.excluded_aggregate_rows,
        "agreed": agreed,
    }
    return ("agree" if agreed else "disagree"), detail


def _has_usable_data(rows: list[dict]) -> bool:
    """True if any cell in any row is non-NULL. An over-constrained filter (or a
    figure that isn't actually a table row) yields rows of all-NULL values, which
    carry no answer."""
    return any(v is not None for row in rows for v in dict(row).values())


async def _run_sql(
    intent: QueryIntent, catalog: dict[str, list[str]]
) -> tuple[list[dict] | None, str, str]:
    """Generate → validate → execute, self-correcting once on a DB or validation
    error. Returns (rows, sql, explanation); rows is None if no query executed."""
    sql = ""
    explanation = ""
    prev_sql: str | None = None
    last_error: str | None = None

    for attempt in range(1, MAX_SQL_ATTEMPTS + 1):
        log.info("Step 2: generating SQL (model=%s, attempt %d)", MODEL, attempt)
        try:
            generated = await _generate_sql(
                intent.clarified_query, catalog, intent.target_table,
                previous_sql=prev_sql, error=last_error,
            )
        except Exception as exc:
            # LLM/API failure during generation (timeout, rate limit, malformed
            # structured output) — treat like a failed attempt rather than
            # crashing the request, so the loop can retry then fall back to
            # textual retrieval, matching the DB-execution failure path below.
            last_error = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            log.info("  SQL generation failed (attempt %d): %s", attempt, last_error)
            continue

        sql = _clean_sql(generated.sql)
        explanation = generated.explanation
        log.info("  generated SQL: %s", sql)

        ok, reason = _validate_select(sql)
        if not ok:
            log.info("  SQL REJECTED: %s", reason)
            prev_sql, last_error = sql, f"rejected: {reason}"
            continue

        try:
            log.info("  executing read-only query")
            async with get_async_session() as session:
                rows = await run_readonly_select(session, sql)
            log.info("  query returned %d row(s)", len(rows))
            return rows, sql, explanation
        except Exception as exc:
            last_error = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            prev_sql = sql
            log.info("  query failed (attempt %d): %s", attempt, last_error)

    return None, sql, explanation


async def run_analytical(intent: QueryIntent) -> SynthesisResult:
    async with get_async_session() as session:
        catalog = await get_schema_catalog(session)

    if not catalog:
        # Document may be text-only — let textual retrieval try.
        log.info("  no tables indexed — falling back to textual retrieval")
        return await run_textual(intent)

    rows, sql, explanation = await _run_sql(intent, catalog)

    if rows and _has_usable_data(rows):
        # Gate 1 (deterministic, primary where it applies): independently recompute
        # the scalar with pandas and compare. Two computations that disagree is a
        # hard red flag — discard the number. This doesn't share the LLM's failure
        # mode, so it's the strongest signal we have for scalar aggregations.
        log.info("Step 2b: deterministic cross-check (pandas recompute)")
        verdict, cross = await _df_cross_check(intent, rows)
        if verdict == "disagree":
            # _df_cross_check always pairs "disagree"/"agree" with a populated
            # detail dict — only "not_applicable" pairs with None.
            assert cross is not None
            log.info(
                "  CROSS-CHECK DISAGREE (sql=%s vs df=%s) — discarding result, "
                "falling back to textual retrieval",
                cross["sql_value"], cross["df_value"],
            )
            return await run_textual(intent)
        log.info("  cross-check: %s", verdict)

        # Gate 2 (LLM, always-on backstop): a clean-running query can still answer
        # the WRONG question (wrong column/filter vs. the question) — which the
        # cross-check can't see, because both sides use the same intent. The
        # faithfulness judge can't catch it either (context IS the result).
        log.info("Step 2c: SQL-correctness eval (model=%s)", SQL_EVAL_MODEL)
        sql_check = await evaluate_sql(
            intent.clarified_query, catalog, sql, explanation, rows
        )
        if not sql_check.correct:
            log.info(
                "  SQL CORRECTNESS eval FAILED: %s — discarding result, "
                "falling back to textual retrieval",
                sql_check.issues,
            )
            return await run_textual(intent)
        log.info("  SQL correctness eval passed")

        # Tier the trust label to the evidence: a cross-check agreement earns the
        # stronger claim; an LLM-only audit is reported as such (don't overstate).
        verification = (
            "independently_confirmed" if verdict == "agree" else "query_audited"
        )
        log.info("  verification tier: %s", verification)

        context = _format_context(sql, explanation, rows, verification)
        sources = [
            SourceRef(
                doc_filename=intent.target_table or "indexed tables",
                page_number=None,
                content_snippet=f"SQL: {sql}",
            )
        ]
        synthesis = await synthesize(
            intent.clarified_query, context, sources, mode="standard"
        )
        # Attach the verified data + how it was verified, checkable without the prose.
        synthesis.result = {
            "kind": "analytical",
            "sql": sql,
            "rows": rows[:100],
            "verification": verification,
            "cross_check": cross,
        }
        if synthesis.eval_passed:
            return synthesis
        log.info(
            "  analytical synthesis did not pass (%s) — falling back to textual retrieval",
            synthesis.answer_basis,
        )
    else:
        n = 0 if not rows else len(rows)
        log.info(
            "  table query yielded no usable data (%d row(s), all NULL/empty) — "
            "falling back to textual retrieval",
            n,
        )

    # Fallback: the figure may be stated in narrative text rather than a table row
    # (e.g. "$1.1B in cash reserves at the end of Q4 2024" appears in prose only).
    return await run_textual(intent)
