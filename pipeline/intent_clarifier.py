"""
Step 1 of the pipeline: classify the user's question and extract structured intent.

The schema catalog (live table + column names from DB) is injected into the system
prompt so the LLM grounds target_table and target_column in actual indexed data.
"""
import logging
from typing import Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.table_store import format_schema_catalog, get_schema_catalog
from database.vector_store import get_narrative_preview

MODEL = "claude-sonnet-5"
log = logging.getLogger("pipeline.intent")


class QueryIntent(BaseModel):
    in_scope: bool = Field(
        default=True,
        description=(
            "FIRST decision: is this question about the indexed financial document(s) — "
            "their figures, tables, narrative, trends, or reasonable predictions grounded "
            "in them? Set False for anything unrelated (general knowledge, other companies, "
            "chit-chat, coding help, etc.). When False, the assistant declines to answer."
        ),
    )
    query_type: Literal["textual", "analytical", "hybrid"] = Field(
        description=(
            "textual: answered by searching document narrative text. "
            "analytical: answered by fetching and computing over table data. "
            "hybrid: requires both document text AND table data, or asks for predictions/trends."
        )
    )
    retrieval_style: Literal["specific", "broad"] = Field(
        default="specific",
        description=(
            "How narrative text retrieval should judge relevance for this question. "
            "'specific': the question asks for a particular fact, figure, date, or named "
            "detail with one expected-answer value (e.g. 'what percentage of revenue was "
            "cloud', 'where will the new office open') — judged by fact-matching precision. "
            "'broad': the question asks for a summary/overview/general assessment (e.g. "
            "'what were the Q4 highlights', 'summarize the quarter'), OR checks whether the "
            "document mentions/discusses a general topic at all rather than naming one "
            "expected value (e.g. 'is there anything about X', 'are there any plans for Y', "
            "'does the report mention Z') — judged by general topical relevance, since "
            "neither has a single fact-matching target the way a specific lookup does. Only "
            "relevant for textual/hybrid queries; ignored for analytical."
        ),
    )
    clarified_query: str = Field(
        description="The user's question rephrased for clarity, preserving original intent exactly."
    )
    target_table: str | None = Field(
        default=None,
        description=(
            "Exact table name from the schema catalog. "
            "Required for analytical and hybrid queries. Must match catalog exactly."
        ),
    )
    target_column: str | None = Field(
        default=None,
        description=(
            "Exact column name from the schema catalog to aggregate over. "
            "Set when the question targets a specific metric column. "
            "Null when the question spans all columns or needs row-level comparison."
        ),
    )
    row_filter: dict | None = Field(
        default=None,
        description=(
            "Pre-filter rows before computation. "
            "Keys and values must be exact strings from the table data, "
            "e.g. {\"<column name>\": \"<exact cell value>\"}. Null for no filter."
        ),
    )
    aggregation: str | None = Field(
        default=None,
        description=(
            "Natural-language description of the computation needed over the table rows. "
            "Examples: 'sum all values', 'find the row with the highest value', "
            "'compare values across all rows', 'list all rows'. "
            "Null for textual queries."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the CLASSIFICATION (0.0–1.0): how clearly you understood "
            "what is being asked and which table/approach applies. This is NOT about "
            "whether a definitive answer exists. Forward-looking predictions and trend "
            "extrapolations grounded in the indexed data should score HIGH when the "
            "classification is clear. Use a value below 0.75 only when the question is "
            "genuinely unintelligible."
        ),
    )
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True when the question is ambiguous or names a period/entity NOT in the "
            "indexed data but that plausibly maps to something that is — e.g. it asks "
            "about a period just outside the range the report covers. When True, set "
            "clarification_question and do not attempt to answer. For questions clearly "
            "unrelated to the documents, leave this False and classify normally."
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description=(
            "A short question asking the user to resolve the ambiguity. Set only when "
            "needs_clarification is True. The system has no conversation memory, so this "
            "must prompt the user to ask a NEW, fully self-contained question next — never "
            "phrase it as something answerable with a short reply (e.g. not \"did you mean "
            "Q3 or Q4?\"). Example: \"That period isn't covered by this report — please "
            "ask again naming one of the periods it does cover.\""
        ),
    )
    reasoning: str = Field(
        description="Brief internal reasoning for this classification. Not shown to the user."
    )
    original_question: str = Field(
        default="",
        description="(Set by the system — leave empty.) The user's verbatim question.",
    )


_STATIC_INSTRUCTIONS = """You are a query classifier for a financial due diligence question-answering
system that answers ONLY questions about the indexed financial document(s) below.

## Step 0 — Scope (decide this FIRST)

Set `in_scope=False` for anything not about these documents — general knowledge,
other companies, current events, chit-chat, coding help, etc. Everything else
(the documents' figures, tables, narrative, trends, and reasonable predictions
grounded in them) is in scope — including a specific, narrowly-worded question
whose answer might live only in the narrative text (see the preview below), not
in any table. Do NOT reject a question as out-of-scope merely because it doesn't
match a table/column — check the narrative preview too before deciding; a
report's prose commonly covers things like new offices/facilities, hiring plans,
management commentary, and forward guidance that never appear in any table.

## Query types (only when in_scope)

- **textual**: The answer lives in the narrative text of the documents (e.g. highlights,
  strategy descriptions, risk factors, qualitative statements). Retrieve via semantic search.

- **analytical**: The answer requires fetching rows from a structured table and computing
  a result (e.g. totals, maximums, comparisons across rows). No narrative text needed.

- **hybrid**: The answer needs both table data AND document narrative, or asks for a
  forward-looking prediction or trend conclusion that must be grounded in both.

## Rules

1. `target_table` and `target_column` MUST exactly match the schema below — character for character.
2. Never invent table or column names not listed in the schema.
3. `aggregation` should describe what computation the pipeline should perform in plain English
   (e.g. "sum all values in the column", "find the row where the column value is highest").
4. Confidence reflects how clearly you understood and classified the question — NOT whether a
   definitive answer exists. A grounded prediction or trend extrapolation is a valid hybrid
   question and scores HIGH confidence when the classification is clear.
5. Clarification: set `needs_clarification=True` with a specific `clarification_question` when
   an in-scope question is ambiguous or names a period/entity not indexed but likely meaning one
   that is — e.g. it asks about a period just outside the range the report covers. Do NOT guess
   in that case; ask. (This is different from out-of-scope: an unrelated question gets
   in_scope=False, not a clarification.)
6. `retrieval_style`: set 'broad' for genuine summary/overview questions AND for questions
   checking whether the document mentions/discusses a general topic at all (e.g. "summarize
   the quarter", "is there anything about X", "are there any plans for Y") — neither has one
   specific expected-answer value. Set 'specific' only when the question asks for one
   particular figure, date, or named value (e.g. "what percentage...", "which department had
   the highest..."). If unsure whether it's a specific-value lookup versus a topic-existence
   check, prefer 'broad'."""


def _build_corpus_context(
    schema_catalog: dict[str, list[str]], narrative_preview: list[str]
) -> str:
    catalog_str = format_schema_catalog(schema_catalog)
    preview_str = (
        "\n".join(f'  - "{p}..."' for p in narrative_preview)
        if narrative_preview
        else "  (no narrative text indexed)"
    )

    return f"""## Available schema (live from indexed documents)

{catalog_str}

## Narrative text preview (for scope/textual-relevance judgment only — the full
## text is retrieved separately at answer time; these are truncated previews of
## the first {len(narrative_preview)} chunk(s) in document order, not the whole narrative)

{preview_str}"""


async def clarify_intent(question: str, session: AsyncSession) -> QueryIntent:
    # Sequential, not concurrent: both reads share one AsyncSession, which
    # isn't safe to use from two coroutines at once.
    schema_catalog = await get_schema_catalog(session)
    narrative_preview = await get_narrative_preview(session)
    log.info("Step 1: clarifying intent (model=%s, %d tables in schema, %d narrative chunks previewed)",
             MODEL, len(schema_catalog), len(narrative_preview))
    corpus_context = _build_corpus_context(schema_catalog, narrative_preview)

    # method="json_schema" → Claude's constrained-decoding structured outputs,
    # which guarantee the response conforms to the schema (vs. the default
    # function-calling mode, where the shape is a strong prior but can drift).
    # Suppressed: ChatAnthropic's other constructor fields use a
    # Field(None, alias=...) style basedpyright's pydantic support doesn't
    # recognise as having a default, so it misreports them as missing — a
    # false positive in the third-party stub, not a real missing-argument bug.
    llm = ChatAnthropic(model=MODEL)  # pyright: ignore[reportCallIssue]
    llm = llm.with_structured_output(QueryIntent, method="json_schema")

    system_prompt = f"{_STATIC_INSTRUCTIONS}\n\n{corpus_context}"

    # with_structured_output()'s return type is generically `dict | BaseModel`
    # regardless of the schema passed in — cast to the schema we know it is.
    intent = cast(
        QueryIntent,
        await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=question)]
        ),
    )
    # Retrieval should use the user's verbatim words, not the LLM's rephrasing
    # (rephrasing can shift the embedding and miss otherwise-relevant chunks).
    intent.original_question = question
    log.info("  reasoning: %s", intent.reasoning)
    return intent
