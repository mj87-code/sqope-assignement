from dataclasses import dataclass
from typing import Literal

# The intent clarifier tags each query's retrieval style (see
# QueryIntent.retrieval_style): "specific" questions ask for a fact/figure —
# gated on the cross-encoder rerank_score, which separates these cleanly.
# "broad" questions ask for a summary/overview — the reranker scores these
# near zero even against clearly relevant content (cross-encoders are trained
# for fact-matching, not topic-summary relevance), so they're gated on cosine
# similarity instead.
RERANK_THRESHOLD = 0.5
SIMILARITY_THRESHOLD = 0.30


@dataclass
class RetrievalEvalResult:
    passed: bool
    reason: str | None
    top_score: float | None


def evaluate_retrieval(
    chunks: list[dict], retrieval_style: Literal["specific", "broad"] = "specific"
) -> RetrievalEvalResult:
    if not chunks:
        return RetrievalEvalResult(
            passed=False,
            reason="No relevant chunks found in the indexed documents.",
            top_score=None,
        )

    if retrieval_style == "broad":
        top_score = max(c["similarity"] for c in chunks)
        threshold = SIMILARITY_THRESHOLD
    else:
        top_score = max(c["rerank_score"] for c in chunks)
        threshold = RERANK_THRESHOLD

    if top_score < threshold:
        return RetrievalEvalResult(
            passed=False,
            reason=(
                f"Top relevance score {top_score:.2f} is below the threshold "
                f"{threshold}. No sufficiently relevant content found. Try "
                "rephrasing the question or naming the topic more specifically — "
                "retrieval scores can vary a lot with phrasing."
            ),
            top_score=top_score,
        )

    return RetrievalEvalResult(passed=True, reason=None, top_score=top_score)
