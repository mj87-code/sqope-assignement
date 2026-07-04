"""
Shared formatting helpers for building synthesis context from retrieved chunks.
Used by the textual and hybrid pipelines.
"""
from pipeline.synthesizer import SourceRef


def format_chunk_blocks(chunks: list[dict]) -> list[str]:
    """Render retrieved chunks as '[Source i — file, page N]\\ncontent' blocks."""
    return [
        f"[Source {i} — {c['doc_filename']}, page {c['page_number']}]\n{c['content']}"
        for i, c in enumerate(chunks, 1)
    ]


def build_sources(chunks: list[dict]) -> list[SourceRef]:
    return [
        SourceRef(
            doc_filename=c["doc_filename"],
            page_number=c.get("page_number"),
            content_snippet=c["content"][:200],
        )
        for c in chunks
    ]
