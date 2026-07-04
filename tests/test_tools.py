"""
Phase 3 tests — tools layer.

Unit tests: tool schema and structure (no DB, no PDF).
Integration tests: tool invocation against a running DB with indexed data.
  Requires: TEST_DATABASE_URL + NOVATECH_PDF (run `make test-integration`).
"""
import os
from unittest.mock import AsyncMock, patch

import pytest

NOVATECH_PDF = os.getenv("NOVATECH_PDF", "")


# ---------------------------------------------------------------------------
# Unit tests — tool metadata and schema
# ---------------------------------------------------------------------------

class TestSearchTool:
    async def test_tool_selects_by_cosine_then_reranks(self):
        from tools.search import search_text_chunks

        fake_results = [{"content": "Revenue grew", "similarity": 0.9}]

        with patch("tools.search._get_embeddings") as mock_emb, \
             patch("tools.search.get_async_session") as mock_session_ctx, \
             patch("tools.search.similarity_search", new_callable=AsyncMock) as mock_search, \
             patch("tools.search.rerank", side_effect=lambda q, c, k: c[:k]) as mock_rerank:

            mock_emb.return_value.embed_query.return_value = [0.1] * 384
            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_search.return_value = fake_results

            result = await search_text_chunks.ainvoke({"query": "revenue", "k": 3})

        assert result == fake_results
        mock_search.assert_called_once()
        # bi-encoder selects k candidates by cosine; the cross-encoder reorders them
        assert mock_search.call_args[0][2] == 3
        mock_rerank.assert_called_once()


class TestReranker:
    def test_reorders_by_cross_encoder_score(self):
        import tools.reranker as rr
        chunks = [{"content": "irrelevant noise"}, {"content": "the real answer"}]

        class FakeModel:
            def predict(self, pairs):
                return [(-3.0 if "irrelevant" in c else 6.0) for _, c in pairs]

        with patch.object(rr, "_get_model", return_value=FakeModel()):
            out = rr.rerank("q", chunks, top_k=2)

        assert out[0]["content"] == "the real answer"
        assert out[0]["rerank_score"] > out[1]["rerank_score"]
        assert 0.0 <= out[0]["rerank_score"] <= 1.0

    def test_raises_when_model_unavailable(self):
        import tools.reranker as rr
        chunks = [{"content": "a"}, {"content": "b"}, {"content": "c"}]
        with patch.object(rr, "_get_model", side_effect=RuntimeError("offline")):
            with pytest.raises(RuntimeError):
                rr.rerank("q", chunks, top_k=2)


class TestGetTableRowsTool:
    async def test_tool_calls_load_table_rows(self):
        from tools.table_rows import get_table_rows

        fake_rows = [{"Department": "R&D", "Headcount Q3": 2450}]

        with patch("tools.table_rows.get_async_session") as mock_session_ctx, \
             patch("tools.table_rows.load_table_rows", new_callable=AsyncMock) as mock_load:

            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_load.return_value = fake_rows

            result = await get_table_rows.ainvoke({"table_name": "Headcount", "row_filter": None})

        assert result == fake_rows
        mock_load.assert_called_once()

    async def test_tool_passes_row_filter(self):
        from tools.table_rows import get_table_rows

        with patch("tools.table_rows.get_async_session") as mock_session_ctx, \
             patch("tools.table_rows.load_table_rows", new_callable=AsyncMock) as mock_load:

            mock_session = AsyncMock()
            mock_session_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_ctx.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_load.return_value = []

            await get_table_rows.ainvoke(
                {"table_name": "Headcount", "row_filter": {"Department": "R&D"}}
            )

        call_args = mock_load.call_args
        assert call_args[1].get("row_filter") == {"Department": "R&D"} or \
               call_args[0][2] == {"Department": "R&D"}


# ---------------------------------------------------------------------------
# Integration tests — require running DB with indexed NovaTech PDF
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSearchToolIntegration:
    def setup_method(self):
        if not NOVATECH_PDF:
            pytest.skip("NOVATECH_PDF not set")

    async def test_search_returns_results_for_revenue_query(self):
        from tools.search import search_text_chunks
        results = await search_text_chunks.ainvoke({"query": "Q4 revenue highlights", "k": 5})
        assert isinstance(results, list)
        assert len(results) > 0
        assert "content" in results[0]
        assert "similarity" in results[0]


@pytest.mark.integration
class TestGetTableRowsIntegration:
    def setup_method(self):
        if not NOVATECH_PDF:
            pytest.skip("NOVATECH_PDF not set")

    async def test_returns_rows_for_known_table(self, db_session):
        from database.table_store import get_all_table_names
        from tools.table_rows import get_table_rows

        table_names = await get_all_table_names(db_session)
        if not table_names:
            pytest.skip("No tables indexed — run indexer first")

        rows = await get_table_rows.ainvoke({"table_name": table_names[0]})
        assert isinstance(rows, list)
        assert len(rows) > 0

    async def test_row_filter_reduces_results(self, db_session):
        from database.table_store import get_all_table_names
        from tools.table_rows import get_table_rows

        table_names = await get_all_table_names(db_session)
        if not table_names:
            pytest.skip("No tables indexed — run indexer first")

        table = table_names[0]
        all_rows = await get_table_rows.ainvoke({"table_name": table})
        if not all_rows:
            pytest.skip("No rows in first table")

        first_key = next(iter(all_rows[0]))
        first_val = all_rows[0][first_key]
        filtered = await get_table_rows.ainvoke(
            {"table_name": table, "row_filter": {first_key: first_val}}
        )
        assert len(filtered) <= len(all_rows)
