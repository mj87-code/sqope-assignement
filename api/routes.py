import logging

from fastapi import APIRouter

from api.schemas import QueryRequest, QueryResponse, QueryResult, SimpleQueryResponse
from api.trace import start_trace
from database.connection import get_async_session
from evals.intent_eval import evaluate_intent
from pipeline.analytical import run_analytical
from pipeline.hybrid import run_hybrid
from pipeline.intent_clarifier import clarify_intent
from pipeline.textual import run_textual

log = logging.getLogger("pipeline.router")
router = APIRouter()


def _respond(
    response: QueryResponse, verbose: bool
) -> QueryResponse | SimpleQueryResponse:
    """Full QueryResponse when verbose=true; otherwise just the answer (or the
    rejection reason when there isn't one) — the common case doesn't need
    query_type/sources/eval_passed/trace/result to just read the answer."""
    if verbose:
        return response
    return SimpleQueryResponse(
        answer=response.answer,
        reason=response.rejection_reason if response.answer is None else None,
    )


@router.post("/query", response_model=QueryResponse | SimpleQueryResponse)
async def query(request: QueryRequest, verbose: bool = False) -> QueryResponse | SimpleQueryResponse:
    trace = start_trace()
    log.info("=" * 60)
    log.info("QUESTION: %s", request.question)

    async with get_async_session() as session:
        intent = await clarify_intent(request.question, session)

    log.info(
        "intent -> in_scope=%s type=%s confidence=%.2f table=%s",
        intent.in_scope, intent.query_type, intent.confidence, intent.target_table,
    )

    # Scope gate: only answer questions about the indexed financial document(s).
    if not intent.in_scope:
        log.info("OUT OF SCOPE: declining")
        return _respond(QueryResponse(
            answer=None,
            query_type=intent.query_type,
            sources=[],
            eval_passed=False,
            answer_basis="out_of_scope",
            rejection_reason=(
                "This assistant only answers questions about the provided financial "
                "document(s)."
            ),
            confidence=intent.confidence,
            trace=trace,
        ), verbose)

    # Clarification: in-scope but ambiguous / out-of-range period.
    if intent.needs_clarification:
        log.info("NEEDS CLARIFICATION: %s", intent.clarification_question)
        return _respond(QueryResponse(
            answer=None,
            query_type=intent.query_type,
            sources=[],
            eval_passed=False,
            answer_basis="needs_clarification",
            rejection_reason=intent.clarification_question or (
                "Your question is ambiguous — please ask again as a fully "
                "self-contained question naming the exact period, table, or metric you mean."
            ),
            confidence=intent.confidence,
            trace=trace,
        ), verbose)

    intent_eval = evaluate_intent(intent)
    if not intent_eval.passed:
        log.info("REJECTED at intent eval: %s", intent_eval.reason)
        return _respond(QueryResponse(
            answer=None,
            query_type=intent.query_type,
            sources=[],
            eval_passed=False,
            answer_basis="insufficient_data",
            rejection_reason=intent_eval.reason,
            confidence=intent.confidence,
            trace=trace,
        ), verbose)

    log.info("routing to %s pipeline", intent.query_type)
    if intent.query_type == "textual":
        synthesis = await run_textual(intent)
    elif intent.query_type == "analytical":
        synthesis = await run_analytical(intent)
    else:
        synthesis = await run_hybrid(intent)

    log.info(
        "RESULT -> eval_passed=%s basis=%s",
        synthesis.eval_passed, synthesis.answer_basis,
    )
    return _respond(QueryResponse(
        answer=synthesis.answer,
        query_type=intent.query_type,
        sources=synthesis.sources,
        eval_passed=synthesis.eval_passed,
        answer_basis=synthesis.answer_basis,
        rejection_reason=synthesis.rejection_reason,
        confidence=intent.confidence,
        trace=trace,
        result=QueryResult(**synthesis.result) if synthesis.result else None,
    ), verbose)

