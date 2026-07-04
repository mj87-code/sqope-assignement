"""
Phase 1 tests — infrastructure layer.
"""
from database.models import TableRow, TextChunk


class TestModels:
    def test_meta_column_maps_to_metadata_in_db(self):
        # Python attribute is 'meta' to avoid SQLAlchemy's reserved 'metadata'
        # but the DB column is still named 'metadata'
        assert "metadata" in TextChunk.__table__.c
        assert "metadata" in TableRow.__table__.c
