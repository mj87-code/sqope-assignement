"""
SQL-correctness eval — audits whether a generated query actually answers the question.

This is distinct from the faithfulness eval. Faithfulness checks the prose answer
against the *context*, but in the analytical path the context IS the query's
result — so a query that targets the wrong column, mis-scopes its filter, or
picks the wrong aggregate produces a number the faithfulness judge then
rubber-stamps (answer matches context, because the context is the wrong number).
The hallucination has moved upstream into the SQL logic.

This gate closes that hole: given the question, the live schema, the generated
SQL, its stated purpose, and the rows it returned, it decides whether the SQL is
the *right* query for the question. On failure the caller discards the number and
falls back to textual retrieval rather than presenting a wrong figure as verified.
"""
from typing import cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from database.table_store import format_rows, format_schema_catalog

# Auditing query logic against intent is a reasoning task, and a false
# "correct" here lets a wrong number ship as VERIFIED. `temperature` is
# deprecated across the whole Claude 5 family (not just Opus), so determinism
# is requested in the prompt instead.
JUDGE_MODEL = "claude-sonnet-5"

_SYSTEM_PROMPT = """You are a SQL-correctness auditor for a financial due-diligence
question-answering system. A wrong number presented as a fact is worse than no
answer, so your job is to catch queries that run cleanly but answer the WRONG
question.

You are given: the QUESTION, the available SCHEMA (tables and their columns), the
generated SQL, its stated PURPOSE, and a sample of the RESULT rows it returned.
All table data lives in `table_rows(table_name, row_data JSONB, row_index)`, where
every PDF column is a JSONB key read as `row_data->>'Column'`.

Decide whether the SQL correctly and completely answers the question. Check:
- TABLE & COLUMN: it reads the table and column(s) that the question is actually
  about, using names that exist in the schema.
- OPERATION: the aggregate matches the question's intent — sum vs. max/min vs.
  average vs. count vs. a row lookup. (e.g. "which department is highest" needs a
  max/ordering that returns the label, not a SUM.)
- FILTER/SCOPE: the WHERE clause matches the question's scope — the right period,
  entity, or segment — and is neither over-constrained (filtering to a value that
  isn't what was asked) nor under-constrained (summing across periods when one was
  requested).
- AGGREGATE ROWS: subtotal/total rows (labels like 'Total ...') are EXCLUDED when
  the question sums or maximises the components, and included only when the
  question explicitly asks for that stated total. A sum that silently includes a
  'Total' row double-counts.
- PLAUSIBILITY: the returned rows are a sensible answer to the question (not empty,
  not all-NULL, not an obviously wrong shape or unit).

Mark correct=false, with specific issues, if ANY check fails OR you are not
confident the query is right. Bias toward flagging: abstaining is safe, a wrong
"verified" number is not. Do NOT recompute the arithmetic yourself or invent
figures — judge the query logic against the question.

Be consistent and deterministic: apply these checks literally and identically
every time, so the same (question, schema, SQL, result) always yields the same
verdict. Do not introduce variation."""


class SqlEvalResult(BaseModel):
    correct: bool = Field(
        description="True only if the SQL is the right query for the question on every check above."
    )
    issues: list[str] = Field(
        description="Specific problems with the query (wrong column, wrong aggregate, "
        "mis-scoped filter, total-row double-count, etc.). Empty if correct=true."
    )


async def evaluate_sql(
    question: str,
    catalog: dict[str, list[str]],
    sql: str,
    explanation: str,
    rows: list[dict],
) -> SqlEvalResult:
    # Suppressed: ChatAnthropic's other constructor fields use a
    # Field(None, alias=...) style basedpyright's pydantic support doesn't
    # recognise as having a default, so it misreports them as missing — a
    # false positive in the third-party stub, not a real missing-argument bug.
    llm = ChatAnthropic(model=JUDGE_MODEL)  # pyright: ignore[reportCallIssue]
    llm = llm.with_structured_output(SqlEvalResult, method="json_schema")

    sample = "\n".join(format_rows(rows)) or "  (no rows)"
    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"SCHEMA:\n{format_schema_catalog(catalog)}\n\n"
        f"GENERATED SQL:\n{sql}\n\n"
        f"STATED PURPOSE:\n{explanation}\n\n"
        f"RESULT ROWS (sample):\n{sample}"
    )

    # with_structured_output()'s return type is generically `dict | BaseModel`
    # regardless of the schema passed in — cast to the schema we know it is.
    return cast(
        SqlEvalResult,
        await llm.ainvoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_content)]
        ),
    )
