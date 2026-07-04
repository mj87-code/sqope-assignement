import logging

from evals.retrieval_eval import evaluate_retrieval
from pipeline.context_format import build_sources, format_chunk_blocks
from pipeline.intent_clarifier import QueryIntent
from pipeline.synthesizer import SynthesisResult, synthesize
from tools.search import search_text_chunks

log = logging.getLogger("pipeline.textual")


async def run_textual(intent: QueryIntent) -> SynthesisResult:
    log.info("Step 2: vector search (k=8)")
    search_query = intent.original_question or intent.clarified_query
    chunks = await search_text_chunks.ainvoke({"query": search_query, "k": 8})

    retrieval = evaluate_retrieval(chunks, intent.retrieval_style)
    log.info("  retrieval: %d chunks, top_score=%s, passed=%s",
             len(chunks), f"{retrieval.top_score:.3f}" if retrieval.top_score else "n/a",
             retrieval.passed)
    if not retrieval.passed:
        return SynthesisResult(
            answer=None,
            sources=[],
            eval_passed=False,
            answer_basis="insufficient_data",
            rejection_reason=retrieval.reason,
        )

    context = _format_context(chunks)
    sources = build_sources(chunks)
    return await synthesize(intent.clarified_query, context, sources, mode="standard")


def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(format_chunk_blocks(chunks))
