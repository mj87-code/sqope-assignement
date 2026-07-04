"""
Eval gate after intent clarification.

Failures return a structured rejection — the pipeline stops and the reason
is surfaced to the caller rather than passing a bad intent downstream.
"""
from dataclasses import dataclass

from pipeline.intent_clarifier import QueryIntent

CONFIDENCE_THRESHOLD = 0.75


@dataclass
class IntentEvalResult:
    passed: bool
    reason: str | None  # None when passed


def evaluate_intent(intent: QueryIntent) -> IntentEvalResult:
    if intent.confidence < CONFIDENCE_THRESHOLD:
        return IntentEvalResult(
            passed=False,
            reason=(
                f"Query confidence too low ({intent.confidence:.2f} < {CONFIDENCE_THRESHOLD}). "
                "Please rephrase or provide more context."
            ),
        )

    if intent.query_type in ("analytical", "hybrid") and not intent.target_table:
        return IntentEvalResult(
            passed=False,
            reason=(
                f"Could not identify a relevant table for this {intent.query_type} query. "
                "Ensure the question refers to data available in the indexed documents."
            ),
        )

    return IntentEvalResult(passed=True, reason=None)
