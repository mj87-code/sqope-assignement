"""
Hybrid pipeline path — predictions, trends, questions needing both text and table data.

Runs both retrievals concurrently. Either source alone is sufficient to proceed;
only fails if both are empty. Uses grounded-inference mode so the LLM can draw
conclusions from the provided data without reaching into training knowledge.
"""
import asyncio
import logging
import math
import re

from database.table_store import format_rows
from pipeline.context_format import build_sources, format_chunk_blocks
from pipeline.intent_clarifier import QueryIntent
from pipeline.synthesizer import SynthesisResult, synthesize
from pipeline.table_compute import _Computed, _try_compute
from tools.search import search_text_chunks
from tools.table_rows import get_table_rows

log = logging.getLogger("pipeline.hybrid")

# Hybrid includes narrative at moderate relevance: the table anchors the
# answer, so supplementary text (e.g. forward-looking guidance) is useful for
# grounding a prediction. Same thresholds and retrieval_style rationale as
# evals/retrieval_eval.
_HYBRID_RERANK_FLOOR = 0.5
_HYBRID_SIMILARITY_FLOOR = 0.30


def _relevant_enough(chunk: dict, retrieval_style: str) -> bool:
    if retrieval_style == "broad":
        return chunk["similarity"] >= _HYBRID_SIMILARITY_FLOOR
    return chunk["rerank_score"] >= _HYBRID_RERANK_FLOOR


_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Prose often restates a figure at a different unit scale than the table column
# ("$1.42 billion" for a column in USD millions) — a match at any common
# thousands-step scaling counts as agreement.
_UNIT_SCALES = (1.0, 1e3, 1e6, 1e-3, 1e-6)


def _numbers_in(text: str) -> list[float]:
    out = []
    for token in _NUMBER_RE.findall(text):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


def _computed_agrees_with_answer(value: float, answer: str) -> bool:
    """False only on a demonstrable contradiction: the answer states at least
    one number and none of them matches the computed value at any common unit
    scaling. An answer with no numbers can't contradict, so it counts as
    agreement — the guard's bias is to keep, and only drop what provably
    conflicts with the prose."""
    numbers = _numbers_in(answer)
    if not numbers:
        return True
    return any(
        math.isclose(n * scale, value, rel_tol=1e-3)
        for n in numbers
        for scale in _UNIT_SCALES
    )


async def run_hybrid(intent: QueryIntent) -> SynthesisResult:
    log.info("Step 2: retrieving text + table rows concurrently")
    search_query = intent.original_question or intent.clarified_query
    # return_exceptions=True: either source failing (embedding error, DB error)
    # must not cancel the sibling task — a single failed source degrades to
    # "no data from that source" rather than crashing the whole request, per
    # this module's "either source alone is sufficient" guarantee above.
    text_result, table_result = await asyncio.gather(
        search_text_chunks.ainvoke({"query": search_query, "k": 8}),
        get_table_rows.ainvoke({
            "table_name": intent.target_table,
            "row_filter": intent.row_filter,
        }),
        return_exceptions=True,
    )
    if isinstance(text_result, Exception):
        log.warning("  text retrieval failed, proceeding without it: %s", text_result)
        chunks = []
    else:
        chunks = text_result
    if isinstance(table_result, Exception):
        log.warning("  table retrieval failed, proceeding without it: %s", table_result)
        rows = []
    else:
        rows = table_result

    chunks = [c for c in chunks if _relevant_enough(c, intent.retrieval_style)]
    has_text = bool(chunks)
    has_table = bool(rows)
    log.info("  has_text=%s (%d chunks), has_table=%s (%d rows)",
             has_text, len(chunks), has_table, len(rows))

    if not has_text and not has_table:
        return SynthesisResult(
            answer=None,
            sources=[],
            eval_passed=False,
            answer_basis="insufficient_data",
            rejection_reason="No relevant text or table data found for this query.",
        )

    computed = _try_compute(rows, intent.target_column, intent.aggregation or "")
    context = _format_hybrid_context(
        chunks=chunks if has_text else [],
        rows=rows,
        column=intent.target_column,
        computed=computed,
    )
    sources = build_sources(chunks) if has_text else []

    synthesis = await synthesize(intent.clarified_query, context, sources, mode="grounded")
    # Attach the verified table data / computed value behind the prediction.
    if has_table:
        computed_payload = None
        if computed is not None:
            # Consistency guard: _try_compute has no notion of semantic row
            # category, so on a mixed table it can aggregate the wrong row set
            # while the synthesizer, reasoning over the raw rows in context,
            # answers correctly. `computed` is documented as machine-checkable
            # truth independent of the prose — a figure that contradicts the
            # prose must be dropped, not shipped as verified.
            if synthesis.answer is None or _computed_agrees_with_answer(
                float(computed.value), synthesis.answer
            ):
                computed_payload = {
                    "operation": computed.operation,
                    "column": intent.target_column,
                    "value": computed.value,
                    "matched_row": computed.matched_row,
                }
            else:
                log.warning(
                    "  computed %s('%s')=%s contradicts the synthesized answer — "
                    "dropping it from the structured result",
                    computed.operation, intent.target_column, computed.value,
                )
        synthesis.result = {
            "kind": "hybrid",
            "rows": rows[:100],
            "computed": computed_payload,
        }
    return synthesis


def _format_hybrid_context(
    chunks: list[dict],
    rows: list[dict],
    column: str | None,
    computed: _Computed | None,
) -> str:
    parts = []

    if chunks:
        parts.append("=== Document Text ===")
        parts.extend(format_chunk_blocks(chunks))
        parts.append("")

    if rows:
        parts.append("=== Table Data ===")
        parts.append(f"Rows ({len(rows)} total):")
        parts.extend(format_rows(rows))

        if computed:
            parts += [
                "",
                "VERIFIED COMPUTED RESULT "
                "(Python-calculated, mathematically exact — do not round or modify):",
                f"  {computed.operation}('{column}') = {computed.value}",
            ]
            if computed.matched_row:
                parts.append(f"  Row with this value: {computed.matched_row}")
            if computed.excluded_aggregate_rows:
                parts.append(
                    f"  ({computed.excluded_aggregate_rows} total/subtotal row(s) "
                    "excluded from this computation to avoid double-counting.)"
                )

    return "\n".join(parts)
