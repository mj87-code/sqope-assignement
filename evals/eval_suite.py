"""
Eval suite — quantified metrics for retrieval, intent classification,
end-to-end answer correctness, and faithfulness/hallucination detection.

Runs against a small, hand-labeled question set with known-correct ground
truth for the indexed NovaTech Q4 2024 report, using real embeddings, the
real reranker, and real Anthropic API calls (no mocks).

Requires:
  - ANTHROPIC_API_KEY set (real LLM calls: intent clarifier, faithfulness judge)
  - TEST_DATABASE_URL pointing at the DB with the NovaTech Q4 2024 report indexed
  - The API running for the end-to-end section (default http://localhost:8000,
    override with API_URL)

Run:
    TEST_DATABASE_URL=postgresql://sqope:<password>@localhost:5432/sqope \\
        python -m evals.eval_suite
"""
import asyncio
import os
import sys
from dataclasses import dataclass, field

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from evals.faithfulness_eval import evaluate_faithfulness
from pipeline.intent_clarifier import clarify_intent
from tools.search import search_text_chunks

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")


def _to_async_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        return self.fbeta(1.0)

    def fbeta(self, beta: float) -> float:
        """F-beta: weights recall `beta` times as much as precision.
        beta=1 is the balanced F1; beta>1 favors recall, beta<1 favors precision."""
        p, r = self.precision, self.recall
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0


def confusion_report(pairs: list[tuple[str, str]]) -> tuple[dict[str, ClassMetrics], float]:
    """pairs: list of (expected_label, predicted_label). Returns per-label
    precision/recall/F1 (one-vs-rest) plus overall accuracy."""
    labels = sorted({e for e, _ in pairs} | {p for _, p in pairs})
    metrics = {label: ClassMetrics(label) for label in labels}
    correct = 0
    for expected, predicted in pairs:
        if expected == predicted:
            correct += 1
            metrics[expected].tp += 1
        else:
            metrics[expected].fn += 1
            metrics[predicted].fp += 1
    accuracy = correct / len(pairs) if pairs else 0.0
    return metrics, accuracy


def print_confusion_report(title: str, pairs: list[tuple[str, str]]) -> None:
    metrics, accuracy = confusion_report(pairs)
    print(f"\n--- {title} ---")
    print(f"{'label':<20} {'precision':>10} {'recall':>10} {'f1':>10}")
    for label, m in metrics.items():
        print(f"{label:<20} {m.precision:>10.2f} {m.recall:>10.2f} {m.f1:>10.2f}")
    print(f"{'accuracy':<20} {accuracy:>10.2f}   ({sum(1 for e, p in pairs if e == p)}/{len(pairs)})")


# ---------------------------------------------------------------------------
# Section 1 — Intent classification (real LLM call per question)
# ---------------------------------------------------------------------------

INTENT_CASES = [
    ("What is the sum of Q3 headcount across all departments?", "analytical"),
    ("Which department had the highest Q4 headcount?", "analytical"),
    ("Compare operating cash flow between Q3 and Q4.", "analytical"),
    ("What percentage of total revenue came from cloud services?", "textual"),
    ("What is management's outlook for revenue growth in Q1 2025?", "textual"),
    ("Where will the new R&D center be located?", "textual"),
    ("Will R&D headcount likely keep growing into 2025?", "hybrid"),
    ("Given the AI Solutions growth trend, what's a reasonable Q1 2025 forecast?", "hybrid"),
    ("What was Apple's revenue last quarter?", "out_of_scope"),
    ("Write me a Python function to sort a list.", "out_of_scope"),
]


async def run_intent_eval(session) -> list[tuple[str, str]]:
    pairs = []
    for question, expected in INTENT_CASES:
        intent = await clarify_intent(question, session)
        predicted = "out_of_scope" if not intent.in_scope else intent.query_type
        pairs.append((expected, predicted))
        print(f"  [{expected:<12} -> {predicted:<12}] {question}")
    return pairs


# ---------------------------------------------------------------------------
# Section 2 — Retrieval (real embedding + real reranker call per question)
# ---------------------------------------------------------------------------

# expected: a substring that must appear in a retrieved chunk for the question
# to count as "found".
RETRIEVAL_CASES = [
    ("What percentage of total revenue came from cloud services?", "48%"),
    ("What is management's outlook for revenue growth in Q1 2025?", "Q1 2025"),
    ("Where will the new R&D center be located?", "Austin"),
    ("What was customer growth quarter-over-quarter?", "12%"),
    ("How much stock did the company repurchase this quarter?", "$25M"),
]

RETRIEVAL_K = 5


async def run_retrieval_eval() -> tuple[float, float]:
    """Returns (recall@k, MRR)."""
    hits = 0
    reciprocal_ranks = []
    for question, expected_substring in RETRIEVAL_CASES:
        chunks = await search_text_chunks.ainvoke({"query": question, "k": RETRIEVAL_K})
        rank = None
        for i, chunk in enumerate(chunks, start=1):
            if expected_substring.lower() in chunk["content"].lower():
                rank = i
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            print(f"  [FOUND @ rank {rank}] {question!r} -> {expected_substring!r}")
        else:
            reciprocal_ranks.append(0.0)
            print(f"  [MISS]          {question!r} -> {expected_substring!r} not in top-{RETRIEVAL_K}")

    recall_at_k = hits / len(RETRIEVAL_CASES)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return recall_at_k, mrr


# ---------------------------------------------------------------------------
# Section 3 — End-to-end answer correctness (real HTTP call to the running API)
# ---------------------------------------------------------------------------

@dataclass
class E2ECase:
    question: str
    expect_answer_basis: str
    # Acceptable alternative phrasings of the same fact (e.g. "1,420" vs
    # "1420") — the answer only needs to contain ONE of these, not all.
    expect_substrings: list[str] = field(default_factory=list)


E2E_CASES = [
    E2ECase("What is the sum of Q3 headcount across all departments?", "indexed_documents", ["8,200", "8200"]),
    E2ECase("Which department had the highest Q4 headcount?", "indexed_documents", ["R&D"]),
    E2ECase("What was Q4 2024 revenue?", "indexed_documents", ["1.42", "1,420", "1420"]),
    E2ECase("What percentage of total revenue came from cloud services?", "indexed_documents", ["48%"]),
    E2ECase("Where will NovaTech open its new R&D center?", "indexed_documents", ["Austin"]),
    E2ECase("What was Apple's projected revenue for 2025?", "out_of_scope", []),
]


async def run_e2e_eval() -> tuple[int, int]:
    correct = 0
    async with httpx.AsyncClient(timeout=180) as client:
        for case in E2E_CASES:
            resp = await client.post(f"{API_URL}/query?verbose=true", json={"question": case.question})
            resp.raise_for_status()
            result = resp.json()
            basis_ok = result["answer_basis"] == case.expect_answer_basis
            answer = (result.get("answer") or "").lower()
            # expect_substrings lists acceptable alternative phrasings of the same
            # fact (e.g. "1,420" vs "1420") — any one of them is sufficient.
            content_ok = any(s.lower() in answer for s in case.expect_substrings) if case.expect_substrings else True
            ok = basis_ok and content_ok
            correct += int(ok)
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {case.question!r}")
            print(f"        expected basis={case.expect_answer_basis!r}, got {result['answer_basis']!r}")
            if case.expect_substrings:
                print(f"        expected substrings {case.expect_substrings} in answer: {content_ok}")
    return correct, len(E2E_CASES)


# ---------------------------------------------------------------------------
# Section 4 — Faithfulness / hallucination detection (real judge LLM call)
# Reuses the exact cases from tests/test_faithfulness_eval.py's regression
# suite, scored as a binary classification task (positive class = "unfaithful",
# since catching a hallucination is the safety-critical direction).
# ---------------------------------------------------------------------------

_FAITH_CONTEXT = """[Source 1 — novatech_q4_2024.pdf, page 3]
Total Revenue for Q4 2024 was $1.42B, up from $1.31B in Q3 2024.

[Source 2 — novatech_q4_2024.pdf, page 7]
Departmental headcount, Q4 2024:
  R&D: 2,620 | Sales: 1,900 | Operations: 1,750 | G&A: 980
"""

FAITHFULNESS_CASES = [
    ("Q4 2024 revenue was $1.42B, and R&D was the largest department at 2,620.", "standard", False),
    ("Q4 2024 revenue was $1.87B.", "standard", True),
    ("The report was prepared by CFO Jane Halloran.", "standard", True),
    ("NovaTech is headquartered in Austin, Texas.", "standard", True),
    ("Q1 2025 revenue was $1.55B.", "grounded", True),
]


async def run_faithfulness_eval() -> list[tuple[str, str]]:
    pairs = []
    for answer, mode, expect_unfaithful in FAITHFULNESS_CASES:
        result = await evaluate_faithfulness(answer, _FAITH_CONTEXT, mode=mode)
        expected = "unfaithful" if expect_unfaithful else "faithful"
        predicted = "faithful" if result.faithful else "unfaithful"
        pairs.append((expected, predicted))
        print(f"  [{expected:<10} -> {predicted:<10}] {answer!r}")
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set — required for real LLM calls.", file=sys.stderr)
        sys.exit(1)
    if not TEST_DATABASE_URL:
        print("TEST_DATABASE_URL not set — point it at the DB with the NovaTech report indexed.", file=sys.stderr)
        sys.exit(1)

    engine = create_async_engine(_to_async_url(TEST_DATABASE_URL))
    factory = async_sessionmaker(engine, expire_on_commit=False)

    print("=" * 70)
    print("SQOPE EVAL SUITE — real calls against the live pipeline")
    print("=" * 70)

    print("\n[1/4] Intent classification")
    async with factory() as session:
        intent_pairs = await run_intent_eval(session)
    print_confusion_report("Intent classification", intent_pairs)

    print("\n[2/4] Retrieval (recall@k, MRR)")
    recall_at_k, mrr = await run_retrieval_eval()
    print(f"\n--- Retrieval (k={RETRIEVAL_K}) ---")
    print(f"recall@{RETRIEVAL_K}: {recall_at_k:.2f}")
    print(f"MRR:       {mrr:.2f}")

    print(f"\n[3/4] End-to-end answer correctness (live API at {API_URL})")
    try:
        e2e_correct, e2e_total = await run_e2e_eval()
        print(f"\n--- End-to-end accuracy ---")
        print(f"accuracy: {e2e_correct / e2e_total:.2f}   ({e2e_correct}/{e2e_total})")
    except httpx.ConnectError:
        print(f"  SKIPPED — could not reach API at {API_URL} (is it running? `make docker-up`)")

    print("\n[4/4] Faithfulness / hallucination detection")
    faith_pairs = await run_faithfulness_eval()
    print_confusion_report("Faithfulness (positive class = unfaithful)", faith_pairs)
    faith_metrics, _ = confusion_report(faith_pairs)
    f2_unfaithful = faith_metrics["unfaithful"].fbeta(2.0)
    print(
        "\nF2 (unfaithful) = "
        f"{f2_unfaithful:.2f} — weighted 2x toward recall, not F1 or F0.5. "
        'The system\'s own design rule is "a missing answer is always preferable '
        'to an incorrect one" (pipeline/synthesizer.py) — for the judge that gates '
        "on this, missing a real hallucination (a false negative on \"unfaithful\") "
        "is worse than over-rejecting a good answer (a false positive), so recall "
        "of catching hallucinations should be weighted above precision."
    )

    await engine.dispose()

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
