"""
Read-only vector similarity search against text_chunks.
All queries are SELECT only — this module never writes to the database.
"""
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def similarity_search(
    session: AsyncSession,
    query_embedding: list[float],
    k: int = 5,
) -> list[dict]:
    """
    Return the top-k text chunks most similar to query_embedding.
    Each result includes similarity score (0–1, higher is more similar).
    """
    embedding_literal = json.dumps(query_embedding)

    sql = text("""
        SELECT
            tc.id::text,
            tc.content,
            tc.page_number,
            tc.chunk_index,
            tc.metadata,
            d.filename AS doc_filename,
            1 - (tc.embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM text_chunks tc
        JOIN documents d ON d.id = tc.document_id
        WHERE tc.embedding IS NOT NULL
        ORDER BY tc.embedding <=> CAST(:embedding AS vector)
        LIMIT :k
    """)

    result = await session.execute(sql, {"embedding": embedding_literal, "k": k})
    return [dict(row._mapping) for row in result]


async def get_narrative_preview(
    session: AsyncSession, max_chunks: int = 30, preview_chars: int = 100
) -> list[str]:
    """
    Return a short, cheap preview (first `preview_chars` of each of the first
    `max_chunks` chunks, in document order) of the indexed narrative text.

    Used to give the intent clarifier a signal of what topics the narrative
    covers — without it, scope decisions are made blind to prose content
    (only the table schema is visible), so a specific fact that's only in the
    narrative (e.g. "where will the new office open") looks indistinguishable
    from a genuinely out-of-scope question. This is a preview for scope
    judgment only, not a retrieval substitute — bounded in size so it stays
    cheap even for a much larger document.
    """
    sql = text("""
        SELECT LEFT(content, :preview_chars) AS preview
        FROM text_chunks
        ORDER BY chunk_index
        LIMIT :max_chunks
    """)
    result = await session.execute(
        sql, {"preview_chars": preview_chars, "max_chunks": max_chunks}
    )
    return [row.preview for row in result]
