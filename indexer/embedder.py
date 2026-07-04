"""
Embeds text chunks and bulk-inserts into text_chunks.
Inserts table rows (no embedding) into table_rows.
"""
import hashlib
import uuid
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy.orm import Session

from database.models import Document, TableRow, TextChunk
from indexer.chunker import TextChunkData
from indexer.pdf_parser import ParsedDocument, ParsedTable

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model: HuggingFaceEmbeddings | None = None


def _get_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embedding_model


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def index_document(
    session: Session,
    path: Path,
    parsed: ParsedDocument,
    chunks: list[TextChunkData],
) -> str:
    """
    Upsert document, embed text chunks, store table rows.
    Skips if already indexed (same file hash). Returns document ID.
    """
    fhash = file_hash(path)

    existing = session.query(Document).filter_by(file_hash=fhash).first()
    if existing:
        print(f"Already indexed: {parsed.filename} (skipping)")
        return str(existing.id)

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, filename=parsed.filename, file_hash=fhash)
    session.add(doc)
    session.flush()

    _insert_chunks(session, doc_id, chunks)
    _insert_table_rows(session, doc_id, parsed.tables)

    session.commit()
    return str(doc_id)


def _insert_chunks(session: Session, doc_id: uuid.UUID, chunks: list[TextChunkData]) -> None:
    if not chunks:
        return

    model = _get_model()
    texts = [c.content for c in chunks]
    embeddings = model.embed_documents(texts)

    rows = [
        TextChunk(
            document_id=doc_id,
            content=chunks[i].content,
            embedding=embeddings[i],
            page_number=chunks[i].page_number,
            chunk_index=chunks[i].chunk_index,
            meta={"headings": chunks[i].headings},
        )
        for i in range(len(chunks))
    ]
    session.bulk_save_objects(rows)


def _insert_table_rows(
    session: Session, doc_id: uuid.UUID, tables: list[ParsedTable]
) -> None:
    rows = [
        TableRow(
            document_id=doc_id,
            table_name=table.name,
            row_data=row_data,
            row_index=idx,
        )
        for table in tables
        for idx, row_data in enumerate(table.rows)
    ]
    if rows:
        session.bulk_save_objects(rows)
