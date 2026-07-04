"""
Usage:
    python -m indexer.cli /path/to/report.pdf [/path/to/report2.pdf ...]
    python -m indexer.cli --dry-run /path/to/report.pdf   # parse & print, no DB write

--dry-run parses the PDF and prints the extracted tables and text chunks without
touching the database — useful for checking parsing quality on any PDF.
"""
import argparse
import sys
from pathlib import Path

from indexer.chunker import chunk_document
from indexer.pdf_parser import parse_pdf


def index_pdf(path: Path) -> None:
    # Imported lazily so --dry-run works without DB drivers/connection configured.
    from database.connection import get_sync_session
    from indexer.embedder import index_document

    print(f"Parsing {path.name}...")
    parsed, dl_doc = parse_pdf(path)

    print(f"  Tables found: {len(parsed.tables)}")

    print("  Chunking (structure-aware)...")
    chunks = chunk_document(dl_doc)
    print(f"  Chunks produced: {len(chunks)}")

    with get_sync_session() as session:
        doc_id = index_document(session, path, parsed, chunks)

    print(f"  Indexed as document {doc_id}")


def inspect_pdf(path: Path) -> None:
    """Parse and print tables + chunks without writing to the database."""
    print(f"\n{'=' * 70}\nFILE: {path.name}\n{'=' * 70}")
    parsed, dl_doc = parse_pdf(path)
    chunks = chunk_document(dl_doc)

    print(f"\nTABLES: {len(parsed.tables)}")
    for ti, table in enumerate(parsed.tables):
        cols = list(table.rows[0].keys()) if table.rows else []
        print(f"\n  [{ti + 1}] name : {table.name!r}")
        print(f"      rows : {len(table.rows)}")
        print(f"      cols : {cols}")
        for row in table.rows[:5]:
            print(f"        - {row}")
        if len(table.rows) > 5:
            print(f"        ... (+{len(table.rows) - 5} more rows)")

    print(f"\nTEXT CHUNKS: {len(chunks)}")
    for ci, chunk in enumerate(chunks[:5]):
        preview = chunk.content[:120].replace("\n", " ")
        headings = " > ".join(chunk.headings) if chunk.headings else "(none)"
        print(f"\n  [{ci + 1}] page {chunk.page_number} | headings: {headings}")
        print(f"      {preview}{'...' if len(chunk.content) > 120 else ''}")
    if len(chunks) > 5:
        print(f"\n  ... (+{len(chunks) - 5} more chunks)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF files for due diligence QA")
    parser.add_argument("paths", nargs="+", type=Path, help="PDF file paths")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print tables/chunks without writing to the database.",
    )
    args = parser.parse_args()

    errors = []
    for path in args.paths:
        if not path.exists():
            print(f"ERROR: File not found: {path}", file=sys.stderr)
            errors.append(path)
            continue
        if path.suffix.lower() != ".pdf":
            print(f"ERROR: Not a PDF: {path}", file=sys.stderr)
            errors.append(path)
            continue
        try:
            if args.dry_run:
                inspect_pdf(path)
            else:
                index_pdf(path)
        except Exception as exc:
            print(f"ERROR processing {path}: {exc}", file=sys.stderr)
            errors.append(path)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
