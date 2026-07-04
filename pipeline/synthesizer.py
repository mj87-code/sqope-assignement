"""
Shared answer generation step — used by textual, analytical, and hybrid pipelines.

The caller formats the context string and builds sources. The synthesizer:
  1. Calls the LLM (standard or grounded-inference mode).
  2. Detects the INSUFFICIENT_DATA sentinel.
  3. Runs faithfulness eval on any non-sentinel answer.
"""
import logging
from dataclasses import dataclass, field
from typing import Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from evals.faithfulness_eval import JUDGE_MODEL, evaluate_faithfulness

MODEL = "claude-sonnet-5"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
log = logging.getLogger("pipeline.synthesizer")

_STANDARD_SYSTEM = """You are a financial due diligence assistant.

A missing answer is always preferable to an incorrect one.

Rules:
- Use ONLY information explicitly present in the provided context.
- Do NOT use training knowledge, infer beyond what is stated, or estimate.
- Do NOT introduce any fact, figure, or name not present in the context.
- If the specific fact the question asks for IS stated in the context, give it —
  even when the context also contains related, superseded, or seemingly
  conflicting material (e.g. a prior resignation, a different body such as a
  supervisory board, or an earlier period). Report exactly what the context
  states; do not refuse merely because nearby passages are about something else.
- Answer in the same language as the question.
- Use INSUFFICIENT_DATA ONLY when the requested fact genuinely does not appear
  anywhere in the context. If unsure whether it is present, it is present —
  quote it. If the answer cannot be found at all, respond with exactly:
  INSUFFICIENT_DATA"""

_GROUNDED_SYSTEM = """You are a financial due diligence assistant answering an analytical/predictive question.

You are given table data (and possibly document text) below. Reason over it.

This is a PREDICTION / trend question — making a projection is expected, not optional:
- Identify the relevant trend in the provided numbers (quote the specific figures, e.g. the Q3→Q4 change).
- State a reasoned, directional prediction that follows from that trend (e.g. "the line item that grew
  fastest between the two reported periods is the most likely to keep rising").
- Frame predictions explicitly as projections grounded in the data — not as recorded facts.
- If you give a projected number, show it as an estimate derived from the trend and state the basis;
  never present an invented future figure as known.
- Use ONLY the provided context (table + text) and the trends within it. Do NOT use outside knowledge.
- Respond with exactly INSUFFICIENT_DATA only if NO relevant data is provided at all. Do NOT refuse
  merely because the future period itself is not in the data — extrapolating it is the task."""


class SynthesizedAnswer(BaseModel):
    answer: str = Field(
        description="The answer, following the rules above — or exactly INSUFFICIENT_DATA."
    )


class SourceRef(BaseModel):
    doc_filename: str
    page_number: int | None
    content_snippet: str


@dataclass
class SynthesisResult:
    answer: str | None
    sources: list[SourceRef]
    eval_passed: bool
    answer_basis: Literal["indexed_documents", "insufficient_data", "eval_rejected"]
    rejection_reason: str | None
    faithfulness_details: dict | None = field(default=None)
    # Machine-checkable verified data behind the answer (SQL + rows / computed value),
    # independent of the prose. None for textual answers with no computation.
    result: dict | None = field(default=None)


async def synthesize(
    question: str,
    context: str,
    sources: list[SourceRef],
    mode: Literal["standard", "grounded"] = "standard",
) -> SynthesisResult:
    system_prompt = _STANDARD_SYSTEM if mode == "standard" else _GROUNDED_SYSTEM

    log.info("Step 3: synthesizing answer (mode=%s, model=%s)", mode, MODEL)
    # Suppressed: ChatAnthropic's other constructor fields use a
    # Field(None, alias=...) style basedpyright's pydantic support doesn't
    # recognise as having a default, so it misreports them as missing — a
    # false positive in the third-party stub, not a real missing-argument bug.
    llm = ChatAnthropic(model=MODEL)  # pyright: ignore[reportCallIssue]
    llm = llm.with_structured_output(SynthesizedAnswer, method="json_schema")
    # with_structured_output()'s return type is generically `dict | BaseModel`
    # regardless of the schema passed in — cast to the schema we know it is.
    response = cast(
        SynthesizedAnswer,
        await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]),
    )
    answer_text = response.answer.strip()

    if INSUFFICIENT_DATA in answer_text:
        log.info("  synthesizer returned INSUFFICIENT_DATA")
        return SynthesisResult(
            answer=None,
            sources=sources,
            eval_passed=False,
            answer_basis="insufficient_data",
            rejection_reason="No relevant information found in the indexed documents for this query.",
        )

    log.info("Step 4: faithfulness eval (model=%s, mode=%s)", JUDGE_MODEL, mode)
    faith = await evaluate_faithfulness(answer_text, context, mode=mode)
    log.info("  faithfulness: faithful=%s", faith.faithful)
    if not faith.faithful:
        log.info("  unsupported claims: %s", faith.unsupported_claims)

    if not faith.faithful:
        return SynthesisResult(
            answer=None,
            sources=sources,
            eval_passed=False,
            answer_basis="eval_rejected",
            rejection_reason="Answer contains claims not supported by the source documents.",
            faithfulness_details={"unsupported_claims": faith.unsupported_claims},
        )

    return SynthesisResult(
        answer=answer_text,
        sources=sources,
        eval_passed=True,
        answer_basis="indexed_documents",
        rejection_reason=None,
    )
