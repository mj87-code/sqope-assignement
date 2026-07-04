"""
Structure-aware chunking via Docling's HierarchicalChunker.
Respects document structure (headings, sections) so related content
stays together — especially important for financial reports.
"""

from dataclasses import dataclass, field
from typing import cast

from docling_core.transforms.chunker.doc_chunk import DocChunk
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.types.doc.document import DoclingDocument

_chunker = HierarchicalChunker()


@dataclass
class TextChunkData:
    content: str
    page_number: int
    chunk_index: int
    headings: list[str] = field(default_factory=list)


def chunk_document(dl_doc: DoclingDocument) -> list[TextChunkData]:
    """
    Split a DoclingDocument into semantically coherent chunks.
    Section headings are preserved in metadata for richer retrieval context.
    """
    chunks: list[TextChunkData] = []

    # HierarchicalChunker.chunk() is stubbed to yield the generic BaseChunk,
    # but always actually yields DocChunk (whose .meta is DocMeta, with
    # .headings) — cast to the concrete type it's documented to produce.
    for idx, raw_chunk in enumerate(_chunker.chunk(dl_doc)):
        chunk = cast(DocChunk, raw_chunk)
        text = chunk.text.strip() if chunk.text else ""
        if not text:
            continue

        page_number = _page_number(chunk)
        headings = list(chunk.meta.headings) if chunk.meta and chunk.meta.headings else []

        chunks.append(
            TextChunkData(
                content=text,
                page_number=page_number,
                chunk_index=idx,
                headings=headings,
            )
        )

    return chunks


def _page_number(chunk: DocChunk) -> int:
    """
    First page the chunk's content appears on, read from item provenance.
    (chunk.meta.origin is the *document* origin — filename/mimetype — and carries
    no page number, so it must not be used here.)
    """
    try:
        for item in chunk.meta.doc_items:
            if item.prov:
                return item.prov[0].page_no
    except (AttributeError, IndexError, TypeError):
        pass
    return 1
