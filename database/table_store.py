"""
Read-only access to structured table data.
All queries are SELECT only — this module never writes to the database.

get_schema_catalog() is a plain function, NOT a LangChain tool.
It is called in the pre-step before the intent clarifier LLM call
so the LLM receives the live schema in its system prompt.
"""
import json
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def _json_safe(value):
    """Coerce DB values (e.g. asyncpg Decimal) to JSON-serialisable types."""
    if isinstance(value, Decimal):
        # Our financial figures are integers or 1-decimal; float is exact here.
        return int(value) if value == value.to_integral_value() else float(value)
    return value


async def get_schema_catalog(session: AsyncSession) -> dict[str, list[str]]:
    """
    Return {table_name: [column_names]} for all indexed tables.
    Column names are discovered from row_data JSONB keys — no fixed schema required.
    """
    sql = text("""
        SELECT table_name, jsonb_object_keys(row_data) AS col
        FROM table_rows
        GROUP BY table_name, col
        ORDER BY table_name, col
    """)
    result = await session.execute(sql)

    catalog: dict[str, list[str]] = {}
    for row in result:
        catalog.setdefault(row.table_name, []).append(row.col)
    return catalog


def format_schema_catalog(catalog: dict[str, list[str]]) -> str:
    """Render {table: [columns]} as 'Table "x":' / '  - "col"' lines for LLM prompts."""
    if not catalog:
        return "  (no tables indexed yet)"
    lines = []
    for table, cols in catalog.items():
        lines.append(f'  Table "{table}":')
        lines.extend(f'    - "{col}"' for col in cols)
    return "\n".join(lines)


def format_rows(rows: list[dict], limit: int = 50, indent: str = "  ") -> list[str]:
    """Render up to `limit` rows as indented dict-repr lines for LLM context/eval prompts."""
    return [f"{indent}{dict(row)}" for row in rows[:limit]]


async def load_table_rows(
    session: AsyncSession,
    table_name: str,
    row_filter: dict | None = None,
    limit: int | None = 100,
) -> list[dict]:
    """
    Return rows from the named table, optionally filtered by exact JSONB key-value match.
    Results are ordered by row_index. Capped at `limit` rows, or unbounded when
    `limit` is None — callers that aggregate/cross-check over the whole table
    (e.g. the analytical cross-check, the hybrid compute path) must pass
    limit=None: a partial fetch would silently corrupt a "mathematically exact"
    computed figure.
    """
    params: dict = {"table_name": table_name}
    limit_clause = ""
    if limit is not None:
        params["limit"] = limit
        limit_clause = "LIMIT :limit"

    if row_filter:
        params["filter"] = json.dumps(row_filter)
        sql = text(f"""
            SELECT row_data
            FROM table_rows
            WHERE table_name = :table_name
              AND row_data @> CAST(:filter AS jsonb)
            ORDER BY row_index
            {limit_clause}
        """)
    else:
        sql = text(f"""
            SELECT row_data
            FROM table_rows
            WHERE table_name = :table_name
            ORDER BY row_index
            {limit_clause}
        """)

    result = await session.execute(sql, params)
    return [dict(row.row_data) for row in result]


async def get_all_table_names(session: AsyncSession) -> list[str]:
    """Return distinct table names from indexed documents."""
    sql = text("SELECT DISTINCT table_name FROM table_rows ORDER BY table_name")
    result = await session.execute(sql)
    return [row.table_name for row in result]


async def run_readonly_select(session: AsyncSession, sql: str) -> list[dict]:
    """
    Execute a single caller-supplied SELECT inside a READ ONLY transaction.

    `SET TRANSACTION READ ONLY` is issued first, so any write (even if the
    statement validation upstream were bypassed) fails at the database — the
    LLM-generated SQL physically cannot mutate data. Returns rows as dicts.
    """
    await session.execute(text("SET TRANSACTION READ ONLY"))
    result = await session.execute(text(sql))
    return [
        {k: _json_safe(v) for k, v in row.items()}
        for row in result.mappings().all()
    ]
