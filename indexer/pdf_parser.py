"""
PDF parsing via Docling.
Returns structured tables (schema-agnostic rows) + the raw DoclingDocument
for structure-aware chunking by chunker.py.
"""
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import DoclingDocument

log = logging.getLogger("indexer.pdf_parser")

# Caption-like lines: "Table 3: ...", "Figure 1 ...", "Exhibit A ...".
_CAPTION_RE = re.compile(r"^\s*(?:table|figure|exhibit)\b", re.IGNORECASE)

# Accounting negatives are written in parentheses: "(500)" means -500.
_PAREN_NEG_RE = re.compile(r"^\((.+)\)$")
# Stripped (with thousands separators) before testing whether a cell is a number.
_CURRENCY_SYMBOLS = "$€£¥"

# When set (in the Docker image), Docling loads its models from this baked-in
# folder instead of fetching them from the HuggingFace Hub at runtime.
_ARTIFACTS_PATH = os.environ.get("DOCLING_ARTIFACTS_PATH")


@dataclass
class ParsedTable:
    name: str
    rows: list[dict] = field(default_factory=list)


@dataclass
class ParsedDocument:
    filename: str
    tables: list[ParsedTable] = field(default_factory=list)


def parse_pdf(path: Path) -> tuple[ParsedDocument, DoclingDocument]:
    """
    Convert a PDF with Docling.
    Returns (ParsedDocument with tables, raw DoclingDocument for chunking).
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    if _ARTIFACTS_PATH:
        pipeline_options.artifacts_path = _ARTIFACTS_PATH

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(str(path))
    dl_doc = result.document

    tables = _extract_tables(dl_doc)
    parsed = ParsedDocument(filename=path.name, tables=tables)
    return parsed, dl_doc


def _extract_tables(doc: DoclingDocument) -> list[ParsedTable]:
    fallback_captions = _reading_order_captions(doc)
    tables = []
    for i, table_item in enumerate(doc.tables):
        name = _table_name(table_item, doc, fallback_captions, i)
        try:
            df = table_item.export_to_dataframe(doc=doc)
        except Exception:
            continue
        if df.empty:
            continue
        rows = _df_to_rows(df)
        tables.append(ParsedTable(name=name, rows=rows))
    return tables


def _reading_order_captions(doc: DoclingDocument) -> dict[str, str]:
    """
    Pair each table with the caption-like text line immediately preceding it in
    reading order. Docling does not always link a table to its caption (only one
    of four tables was linked in the NovaTech report), so this is the fallback
    that recovers descriptive names like "Table 3: Departmental Headcount".
    """
    mapping: dict[str, str] = {}
    last_caption: str | None = None
    for item, _level in doc.iterate_items():
        if type(item).__name__ == "TableItem":
            if last_caption:
                mapping[item.self_ref] = last_caption
            last_caption = None  # consume so it isn't reused for the next table
        else:
            text = (getattr(item, "text", "") or "").strip()
            if text and _CAPTION_RE.match(text):
                last_caption = text
    return mapping


def _table_name(
    table_item, doc: DoclingDocument, fallback: dict[str, str], index: int
) -> str:
    # 1. Docling's own caption association (most reliable when present).
    try:
        native = table_item.caption_text(doc)
        if native and native.strip():
            return native.strip()
    except Exception:
        pass
    # 2. Reading-order fallback for tables Docling didn't link.
    ref = getattr(table_item, "self_ref", None)
    if ref and ref in fallback:
        return fallback[ref]
    # 3. Generic name — keeps queries working even with no caption at all.
    return f"Table {index + 1}"


def _is_blank(raw) -> bool:
    """True only for a genuinely empty cell (None, NaN, or whitespace)."""
    if raw is None:
        return True
    if isinstance(raw, float) and pd.isna(raw):
        return True
    return str(raw).strip() == ""


def _coerce_number(raw):
    """
    Parse one cell to a native int/float, or return None if it is NOT a pure number.

    Handles the forms real financial tables use and that naive `to_numeric` drops:
      - accounting negatives in parentheses: "(500)" -> -500, "($1,200)" -> -1200
      - thousands separators and currency symbols: "$1,420" -> 1420
    Percentages ("12%") deliberately return None so the caller keeps them as text —
    coercing "12%" to the bare number 12 would let a later SUM/AVG treat a rate as
    a raw quantity, which is a worse error than leaving it as a string.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return None if (isinstance(raw, float) and pd.isna(raw)) else raw
    if raw is None:
        return None

    s = str(raw).strip()
    if not s or s.endswith("%"):
        return None

    m = _PAREN_NEG_RE.match(s)
    negative = m is not None
    body = m.group(1).strip() if m else s

    cleaned = body
    for sym in _CURRENCY_SYMBOLS:
        cleaned = cleaned.replace(sym, "")
    cleaned = cleaned.replace(",", "").strip()

    try:
        num = float(cleaned)
    except ValueError:
        return None
    if negative:
        num = -num
    return int(num) if num.is_integer() else num


def _df_to_rows(df: pd.DataFrame) -> list[dict]:
    """
    Convert a parsed table to JSONB-ready rows, faithfully.

    A column is treated as numeric only if a majority of its non-blank cells parse
    as numbers (so label columns stay text). In a numeric column, genuine numbers
    are normalised via `_coerce_number` (parentheses negatives, currency, thousands
    separators); a cell that still doesn't parse (e.g. "N/A") is KEPT as its
    original string, never nulled. Only genuinely empty cells become None.

    This replaces a column-wide `pd.to_numeric(errors="coerce")` that silently
    turned accounting negatives "(500)", percentages, and stray labels into
    NaN → None — dropping real figures and biasing every downstream SUM/AVG.
    """
    records = df.to_dict(orient="records")

    numeric_cols: set = set()
    for col in df.columns:
        nonblank = [r.get(col) for r in records if not _is_blank(r.get(col))]
        if nonblank and sum(_coerce_number(c) is not None for c in nonblank) >= len(nonblank) / 2:
            numeric_cols.add(col)

    rows: list[dict] = []
    for record in records:
        out: dict = {}
        for key, raw in record.items():
            if _is_blank(raw):
                out[key] = None
                continue
            value = _coerce_number(raw) if key in numeric_cols else None
            # Faithful fallback: a non-blank cell that isn't a number stays as its
            # exact string. It must never silently vanish to None.
            out[key] = value if value is not None else str(raw).strip()
        rows.append(out)

    # Safety assertion: no non-empty source cell may have been dropped to None.
    for record, out in zip(records, rows):
        for key, raw in record.items():
            if not _is_blank(raw) and out[key] is None:
                log.warning(
                    "non-empty cell %r in column %r coerced to None — restoring original",
                    raw, key,
                )
                out[key] = str(raw).strip()

    return rows
