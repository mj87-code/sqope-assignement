from langchain_core.tools import tool

from database.connection import get_async_session
from database.table_store import load_table_rows


@tool
async def get_table_rows(
    table_name: str,
    row_filter: dict | None = None,
) -> list[dict]:
    """
    Retrieve rows from an indexed table.
    Optionally filter by exact column value(s) using row_filter (e.g. {"Department": "R&D"}).
    Returns all matching rows ordered by row_index.
    """
    async with get_async_session() as session:
        # limit=None: the hybrid pipeline computes a "mathematically exact"
        # aggregate over these rows — a partial fetch would silently corrupt it.
        # Display/response size is bounded downstream (format_rows, rows[:100]).
        return await load_table_rows(session, table_name, row_filter, limit=None)
