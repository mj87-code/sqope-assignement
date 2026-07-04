"""
Phase 2 tests — indexer.

Unit tests run without DB or real PDFs (mock-based).
Integration tests require:
  - TEST_DATABASE_URL set
  - NOVATECH_PDF path set to the NovaTech Q4 report PDF
"""
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

NOVATECH_PDF = os.getenv("NOVATECH_PDF", "")


# ---------------------------------------------------------------------------
# Unit tests — no DB, no real PDF
# ---------------------------------------------------------------------------

class TestDfToRows:
    def test_numeric_coercion(self):
        import pandas as pd

        from indexer.pdf_parser import _df_to_rows

        df = pd.DataFrame({"Dept": ["R&D", "Sales"], "Headcount": ["2,450", "1,850"]})
        rows = _df_to_rows(df)
        assert rows[0]["Headcount"] == 2450.0
        assert rows[1]["Headcount"] == 1850.0

    def test_non_numeric_stays_as_string(self):
        import pandas as pd

        from indexer.pdf_parser import _df_to_rows

        df = pd.DataFrame({"Department": ["R&D", "Sales", "Ops"]})
        rows = _df_to_rows(df)
        assert all(isinstance(r["Department"], str) for r in rows)

    def test_nan_becomes_none(self):
        import pandas as pd

        from indexer.pdf_parser import _df_to_rows

        df = pd.DataFrame({"Val": [1.0, float("nan")]})
        rows = _df_to_rows(df)
        assert rows[1]["Val"] is None

    def test_accounting_negative_in_parentheses(self):
        # "(500)" is accounting notation for -500 — must NOT be dropped to None.
        import pandas as pd

        from indexer.pdf_parser import _df_to_rows

        df = pd.DataFrame({"Cash Flow": ["1,200", "(500)", "($1,200)"]})
        rows = _df_to_rows(df)
        assert rows[0]["Cash Flow"] == 1200
        assert rows[1]["Cash Flow"] == -500
        assert rows[2]["Cash Flow"] == -1200

    def test_currency_symbols_stripped(self):
        import pandas as pd

        from indexer.pdf_parser import _df_to_rows

        df = pd.DataFrame({"Revenue": ["$1,420", "$1,310"]})
        rows = _df_to_rows(df)
        assert rows[0]["Revenue"] == 1420
        assert rows[1]["Revenue"] == 1310

    def test_percentages_preserved_as_string_not_nulled(self):
        # "12%" must not vanish, and must not become the bare number 12 (which a
        # later SUM/AVG would mis-add). It stays as the exact string.
        import pandas as pd

        from indexer.pdf_parser import _df_to_rows

        df = pd.DataFrame({"Margin": ["12%", "15%", "9%"]})
        rows = _df_to_rows(df)
        assert [r["Margin"] for r in rows] == ["12%", "15%", "9%"]

    def test_non_numeric_cell_in_numeric_column_kept_not_dropped(self):
        # A stray "N/A" in an otherwise-numeric column must survive as its
        # original string, not be silently dropped to None.
        import pandas as pd

        from indexer.pdf_parser import _df_to_rows

        df = pd.DataFrame({"Headcount": ["2,450", "N/A", "1,850"]})
        rows = _df_to_rows(df)
        assert rows[0]["Headcount"] == 2450
        assert rows[1]["Headcount"] == "N/A"
        assert rows[2]["Headcount"] == 1850

    def test_no_nonblank_cell_becomes_none(self):
        # The core invariant: every non-empty source cell is preserved (as number
        # or string); only genuinely empty cells are None.
        import pandas as pd

        from indexer.pdf_parser import _df_to_rows

        df = pd.DataFrame({
            "Item": ["Revenue", "COGS", ""],
            "Q4": ["1,420", "(500)", "12%"],
        })
        rows = _df_to_rows(df)
        for record, row in zip(df.to_dict(orient="records"), rows, strict=True):
            for key, raw in record.items():
                if str(raw).strip():
                    assert row[key] is not None, f"{raw!r} in {key} was dropped"


class TestChunkDocument:
    def test_chunk_document_returns_list(self):
        from indexer.chunker import chunk_document

        mock_doc = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Revenue grew 14% year over year driven by cloud and AI."
        mock_chunk.meta.headings = ["Q4 Financial Highlights"]
        # Page number comes from item provenance, not the document origin.
        mock_chunk.meta.doc_items = [MagicMock(prov=[MagicMock(page_no=3)])]

        with patch("indexer.chunker._chunker") as mock_chunker:
            mock_chunker.chunk.return_value = [mock_chunk]
            result = chunk_document(mock_doc)

        assert len(result) == 1
        assert result[0].content == "Revenue grew 14% year over year driven by cloud and AI."
        assert result[0].page_number == 3
        assert result[0].headings == ["Q4 Financial Highlights"]

    def test_empty_chunks_are_skipped(self):
        from indexer.chunker import chunk_document

        mock_doc = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "   "  # whitespace only

        with patch("indexer.chunker._chunker") as mock_chunker:
            mock_chunker.chunk.return_value = [mock_chunk]
            result = chunk_document(mock_doc)

        assert result == []

    def test_missing_meta_uses_page_1(self):
        from indexer.chunker import chunk_document

        mock_doc = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.text = "Some content"
        mock_chunk.meta = None

        with patch("indexer.chunker._chunker") as mock_chunker:
            mock_chunker.chunk.return_value = [mock_chunk]
            result = chunk_document(mock_doc)

        assert result[0].page_number == 1


# ---------------------------------------------------------------------------
# Integration tests — require NOVATECH_PDF + TEST_DATABASE_URL
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIndexDocumentDB:
    def setup_method(self):
        if not NOVATECH_PDF:
            pytest.skip("NOVATECH_PDF not set")

    async def test_index_creates_document_record(self, db_session):
        from sqlalchemy import text

        from database.connection import get_sync_session
        from indexer.chunker import chunk_document
        from indexer.embedder import index_document
        from indexer.pdf_parser import parse_pdf

        path = Path(NOVATECH_PDF)
        parsed, dl_doc = parse_pdf(path)
        chunks = chunk_document(dl_doc)

        with get_sync_session() as session:
            doc_id = index_document(session, path, parsed, chunks)

        result = await db_session.execute(
            text("SELECT id FROM documents WHERE id = :id"),
            {"id": doc_id},
        )
        assert result.scalar() is not None

    async def test_index_is_idempotent(self, db_session):
        from sqlalchemy import text

        from database.connection import get_sync_session
        from indexer.chunker import chunk_document
        from indexer.embedder import index_document
        from indexer.pdf_parser import parse_pdf

        path = Path(NOVATECH_PDF)
        parsed, dl_doc = parse_pdf(path)
        chunks = chunk_document(dl_doc)

        with get_sync_session() as session:
            id1 = index_document(session, path, parsed, chunks)
        with get_sync_session() as session:
            id2 = index_document(session, path, parsed, chunks)

        assert id1 == id2

        result = await db_session.execute(
            text("SELECT COUNT(*) FROM documents WHERE filename = :fn"),
            {"fn": parsed.filename},
        )
        assert result.scalar() == 1
