"""
Phase 8 tests — FastAPI routes and end-to-end pipeline.

Unit tests mock the pipeline internals — no DB, no LLM.
Integration tests run the full stack against a live DB with the NovaTech PDF indexed.
  Requires: ANTHROPIC_API_KEY + TEST_DATABASE_URL + NOVATECH_PDF indexed.
"""
import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

API_BASE = os.getenv("API_URL", "http://localhost:8000")


def _make_app():
    """Build app without running lifespan (for unit tests)."""
    from fastapi import FastAPI

    from api.routes import router
    app = FastAPI()
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# Route unit tests (mocked pipeline — no DB, no LLM)
# ---------------------------------------------------------------------------

class TestQueryRoute:
    def _mock_intent(self, query_type="textual", confidence=0.9, **kwargs):
        from pipeline.intent_clarifier import QueryIntent
        return QueryIntent(
            query_type=query_type,
            clarified_query="What were the Q4 highlights?",
            entities=["Q4"],
            confidence=confidence,
            reasoning="test",
            **kwargs,
        )

    def _session_patch(self, mock_session_ctx):
        from unittest.mock import AsyncMock
        mock_session = AsyncMock()
        mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

    def _mock_synthesis(self, passed=True):
        from pipeline.synthesizer import SourceRef, SynthesisResult
        if passed:
            return SynthesisResult(
                answer="Q4 revenue was $1.42B.",
                sources=[SourceRef(doc_filename="novatech.pdf", page_number=5, content_snippet="Revenue...")],
                eval_passed=True,
                answer_basis="indexed_documents",
                rejection_reason=None,
            )
        return SynthesisResult(
            answer=None,
            sources=[],
            eval_passed=False,
            answer_basis="insufficient_data",
            rejection_reason="No data found.",
        )

    async def test_out_of_scope_declined(self):
        app = _make_app()
        intent = self._mock_intent(in_scope=False)

        with patch("api.routes.get_async_session") as mock_session_ctx, \
             patch("api.routes.clarify_intent", new_callable=AsyncMock, return_value=intent), \
             patch("api.routes.run_textual", new_callable=AsyncMock) as mock_textual:
            self._session_patch(mock_session_ctx)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/query?verbose=true", json={"question": "Who is the president of France?"})

        body = resp.json()
        assert body["answer_basis"] == "out_of_scope"
        assert body["answer"] is None
        assert body["rejection_reason"]
        mock_textual.assert_not_called()

    async def test_needs_clarification_returned(self):
        app = _make_app()
        intent = self._mock_intent(
            query_type="hybrid",
            needs_clarification=True,
            clarification_question="'Q5' isn't a standard quarter — did you mean Q1 2025?",
        )

        with patch("api.routes.get_async_session") as mock_session_ctx, \
             patch("api.routes.clarify_intent", new_callable=AsyncMock, return_value=intent), \
             patch("api.routes.run_hybrid", new_callable=AsyncMock) as mock_hybrid:
            self._session_patch(mock_session_ctx)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/query?verbose=true", json={"question": "predict Q5"})

        body = resp.json()
        assert body["answer_basis"] == "needs_clarification"
        assert "Q1 2025" in body["rejection_reason"]
        mock_hybrid.assert_not_called()

    async def test_low_confidence_returns_rejection(self):
        app = _make_app()
        intent = self._mock_intent(confidence=0.5)

        from evals.intent_eval import IntentEvalResult
        rejection = IntentEvalResult(passed=False, reason="Confidence too low.")

        with patch("api.routes.get_async_session") as mock_session_ctx, \
             patch("api.routes.clarify_intent", new_callable=AsyncMock, return_value=intent), \
             patch("api.routes.evaluate_intent", return_value=rejection):

            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                resp = await client.post("/query?verbose=true", json={"question": "???"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["eval_passed"] is False
        assert body["answer"] is None
        assert "confidence" in body

    async def test_analytical_routes_to_run_analytical(self):
        app = _make_app()
        intent = self._mock_intent(query_type="analytical")
        intent.target_table = "Headcount"
        synthesis = self._mock_synthesis()

        from evals.intent_eval import IntentEvalResult

        with patch("api.routes.get_async_session") as mock_session_ctx, \
             patch("api.routes.clarify_intent", new_callable=AsyncMock, return_value=intent), \
             patch("api.routes.evaluate_intent", return_value=IntentEvalResult(passed=True, reason=None)), \
             patch("api.routes.run_analytical", new_callable=AsyncMock, return_value=synthesis) as mock_analytical, \
             patch("api.routes.run_textual", new_callable=AsyncMock) as mock_textual, \
             patch("api.routes.run_hybrid", new_callable=AsyncMock) as mock_hybrid:

            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                await client.post("/query", json={"question": "Sum headcount Q3"})

        mock_analytical.assert_called_once()
        mock_textual.assert_not_called()
        mock_hybrid.assert_not_called()

    async def test_hybrid_routes_to_run_hybrid(self):
        app = _make_app()
        intent = self._mock_intent(query_type="hybrid")
        intent.target_table = "Headcount"
        synthesis = self._mock_synthesis()

        from evals.intent_eval import IntentEvalResult

        with patch("api.routes.get_async_session") as mock_session_ctx, \
             patch("api.routes.clarify_intent", new_callable=AsyncMock, return_value=intent), \
             patch("api.routes.evaluate_intent", return_value=IntentEvalResult(passed=True, reason=None)), \
             patch("api.routes.run_hybrid", new_callable=AsyncMock, return_value=synthesis) as mock_hybrid, \
             patch("api.routes.run_textual", new_callable=AsyncMock), \
             patch("api.routes.run_analytical", new_callable=AsyncMock):

            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                await client.post("/query", json={"question": "predict Q1 hiring"})

        mock_hybrid.assert_called_once()


# ---------------------------------------------------------------------------
# End-to-end integration tests — all 5 assignment queries
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEndToEnd:
    """
    Full pipeline: POST /query → intent → pipeline → response.
    Requires running DB with NovaTech PDF indexed and ANTHROPIC_API_KEY set.
    """

    def setup_method(self):
        try:
            resp = httpx.get(f"{API_BASE}/health", timeout=5.0)
            resp.raise_for_status()
        except Exception:
            pytest.skip(f"API not reachable at {API_BASE} — start docker-compose first")

    async def _post(self, question: str) -> dict:
        async with httpx.AsyncClient(base_url=API_BASE) as client:
            # Generous timeout: a request makes 2–3 sequential LLM calls and can be
            # slow during Anthropic API latency spikes; queries normally take ~10–20s.
            resp = await client.post("/query?verbose=true", json={"question": question}, timeout=240.0)
        assert resp.status_code == 200, resp.text
        return resp.json()

    # --- Query 1: textual ---
    async def test_q1_q4_highlights_textual(self):
        body = await self._post("What were the main Q4 highlights?")
        assert body["query_type"] == "textual"
        assert body["confidence"] >= 0.75
        # Must actually retrieve and answer — guards against a broken vector index
        # or a miscalibrated retrieval threshold silently returning nothing.
        assert body["eval_passed"] is True, body.get("rejection_reason")
        assert body["answer"] is not None
        assert len(body["sources"]) > 0

    async def test_textual_expansion_plans_found(self):
        """The 'Austin R&D / hiring expansion' note must be retrievable."""
        body = await self._post("Are there any expansion plans?")
        assert body["query_type"] in ("textual", "hybrid")
        assert body["eval_passed"] is True, body.get("rejection_reason")
        assert body["answer"] is not None

    # --- Query 2: analytical sum ---
    async def test_q2_total_headcount_q3(self):
        body = await self._post("What was the total headcount across all departments in Q3?")
        assert body["query_type"] == "analytical"
        assert body["confidence"] >= 0.75
        if body["eval_passed"]:
            answer_clean = body["answer"].replace(",", "")
            assert "8200" in answer_clean, f"Expected 8200 in: {body['answer']}"

    # --- Query 3: analytical max ---
    async def test_q3_highest_headcount_dept_q4(self):
        body = await self._post("Which department had the highest headcount in Q4?")
        assert body["query_type"] == "analytical"
        assert body["confidence"] >= 0.75
        if body["eval_passed"]:
            assert "r&d" in body["answer"].lower() or "research" in body["answer"].lower(), \
                f"Expected R&D in: {body['answer']}"

    # --- Query 4: analytical/hybrid comparison ---
    async def test_q4_headcount_q3_vs_q4(self):
        body = await self._post("How did Q3 headcount compare to Q4 across departments?")
        assert body["query_type"] in ("analytical", "hybrid")
        assert body["confidence"] >= 0.75

    # --- Query 5: hybrid prediction ---
    async def test_q5_hiring_prediction(self):
        body = await self._post(
            "Based on the headcount trends, which department is likely to need the most hiring in Q1?"
        )
        assert body["query_type"] == "hybrid"
        assert body["confidence"] >= 0.75
        if body["eval_passed"]:
            assert body["answer"] is not None

    async def test_out_of_scope_response_shape(self):
        """A question unrelated to the financial document should be declined cleanly."""
        body = await self._post("What is the prime minister of France?")
        assert body["answer_basis"] in ("out_of_scope", "insufficient_data", "eval_rejected")
        assert body["answer"] is None
        assert body["rejection_reason"] is not None
