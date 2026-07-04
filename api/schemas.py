from typing import Literal

from pydantic import BaseModel, Field

from pipeline.synthesizer import SourceRef


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, description="The due diligence question to answer.")


class QueryResult(BaseModel):
    """
    Machine-checkable verified data behind the answer — independent of the prose.
    A reviewer can assert the exact figure here without parsing the answer text.
    """
    kind: str = Field(description="Which pipeline produced it: 'analytical' or 'hybrid'.")
    sql: str | None = Field(
        default=None, description="The executed read-only SQL (analytical queries)."
    )
    rows: list[dict] | None = Field(
        default=None, description="Raw DB/query result rows used to produce the answer."
    )
    computed: dict | None = Field(
        default=None,
        description="Python-computed aggregate (hybrid): operation, column, value, matched_row.",
    )
    verification: Literal["independently_confirmed", "query_audited"] | None = Field(
        default=None,
        description=(
            "How strongly the analytical figure was verified. "
            "independently_confirmed: the SQL result was cross-checked by an independent "
            "pandas recomputation and the two agree (strongest). "
            "query_audited: query logic was LLM-audited but not independently recomputed "
            "(no scalar cross-check applied, e.g. a comparison/grouping query)."
        ),
    )
    cross_check: dict | None = Field(
        default=None,
        description="Cross-check detail when applicable: sql_value, df_value, operation, agreed.",
    )


class SimpleQueryResponse(BaseModel):
    """Default /query response shape — just the answer. Pass ?verbose=true for
    the full QueryResponse (query_type, sources, eval_passed, trace, etc.)."""
    answer: str | None = Field(
        description="The answer, or null when no answer was produced (see `reason`)."
    )
    reason: str | None = Field(
        default=None,
        description="Set when answer is null — why no answer was given.",
    )


class QueryResponse(BaseModel):
    answer: str | None = Field(
        description="The answer, or null when eval_passed=false."
    )
    query_type: str = Field(
        description="Classified query type: textual, analytical, or hybrid."
    )
    sources: list[SourceRef] = Field(
        description="Source references used to produce the answer."
    )
    eval_passed: bool = Field(
        description="True if the answer passed all eval gates."
    )
    answer_basis: Literal[
        "indexed_documents",
        "insufficient_data",
        "eval_rejected",
        "out_of_scope",
        "needs_clarification",
    ] = Field(
        description=(
            "indexed_documents: grounded answer from sources. "
            "insufficient_data: no relevant content found. "
            "eval_rejected: answer generated but failed an eval gate. "
            "out_of_scope: question is not about the indexed financial document(s). "
            "needs_clarification: in-scope but ambiguous; rejection_reason holds the clarifying question."
        )
    )
    rejection_reason: str | None = Field(
        description="Human-readable explanation when eval_passed=false."
    )
    confidence: float = Field(
        description="Intent clarifier confidence (0–1). Always present."
    )
    trace: list[str] = Field(
        default_factory=list,
        description="Ordered pipeline step log for this request (intent, routing, retrieval/SQL, synthesis, evals).",
    )
    result: QueryResult | None = Field(
        default=None,
        description=(
            "Structured verified data behind the answer (executed SQL + rows, or computed "
            "value). Null for textual answers and for rejections with no computed data."
        ),
    )
