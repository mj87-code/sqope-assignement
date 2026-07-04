"""
LLM-as-judge faithfulness eval.
Uses claude-sonnet-5 to check that every claim in the answer is supported by the context.
Runs after every answer generation — the safety gate against subtle hallucination.
"""
from typing import cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# The `temperature` param is deprecated across the whole Claude 5 family
# (Opus 4.8, Sonnet 5) — not just Opus — so determinism is requested in the
# prompt instead (see the "Be consistent" instruction in the prompts below)
# rather than set on the model.
JUDGE_MODEL = "claude-sonnet-5"

_STANDARD_PROMPT = """You are a faithfulness evaluator for a financial due diligence system.

Decide whether the ANSWER is supported by the CONTEXT.

Rules:
- Every figure, percentage, date, and proper noun in the answer must be present in the context.
  Formatting may differ — "$5M" and "5" (in millions) match; "12% QoQ" and "12% quarter-over-quarter" match.
- Paraphrase, reordering, and summarising are acceptable — judge the substance, not the wording.
- Flag a claim ONLY if it states a fact, number, or name that is NOT in the context, or relies on
  outside/world knowledge.
- Be consistent and deterministic: apply these rules literally and identically every time, so the
  same (answer, context) always yields the same verdict. Do not introduce variation.
"""

# Grounded answers (predictions, trend summaries, "how was the quarter") are
# allowed to reason over the data, so the judge must not flag legitimate
# conclusions — only fabricated specifics or outside knowledge.
_GROUNDED_PROMPT = """You are a faithfulness evaluator for a financial due diligence system,
checking an analytical/predictive answer that is ALLOWED to reason over the provided data.

Decide whether the ANSWER stays grounded in the CONTEXT.

Acceptable — do NOT flag:
- Reasoned conclusions, comparisons, trends, and qualitative summaries derived from the
  numbers/text in the context (e.g. "revenue grew", "a strong quarter", "this segment is the largest").
- Forward-looking projections explicitly framed as estimates extrapolated from the shown figures.

Flag as unsupported ONLY:
- A specific figure, percentage, date, or proper noun that does NOT appear in the context.
- Any claim requiring outside/world knowledge beyond the provided data.
- A projected number presented as a recorded fact.

Be consistent and deterministic: apply these rules literally and identically every time, so the
same (answer, context) always yields the same verdict. Do not introduce variation.
"""


class FaithfulnessResult(BaseModel):
    faithful: bool = Field(
        description="True only if every factual claim in the answer is supported by the context."
    )
    unsupported_claims: list[str] = Field(
        description="List of specific claims in the answer that are not supported by the context. Empty if faithful=true."
    )


async def evaluate_faithfulness(
    answer: str, context: str, mode: str = "standard"
) -> FaithfulnessResult:
    system_prompt = _GROUNDED_PROMPT if mode == "grounded" else _STANDARD_PROMPT
    # Suppressed: ChatAnthropic's other constructor fields use a
    # Field(None, alias=...) style basedpyright's pydantic support doesn't
    # recognise as having a default, so it misreports them as missing — a
    # false positive in the third-party stub, not a real missing-argument bug.
    llm = ChatAnthropic(model=JUDGE_MODEL)  # pyright: ignore[reportCallIssue]
    llm = llm.with_structured_output(FaithfulnessResult, method="json_schema")

    user_content = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"

    # with_structured_output()'s return type is generically `dict | BaseModel`
    # regardless of the schema passed in — cast to the schema we know it is.
    return cast(
        FaithfulnessResult,
        await llm.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        ),
    )
