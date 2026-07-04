"""
Keyword-guided pandas computation over fetched table rows.

Used by the hybrid pipeline to give the LLM a Python-verified number to ground a
prediction. The analytical pipeline uses text-to-SQL instead (see analytical.py).
"""
import re
from dataclasses import dataclass

import pandas as pd

# Generic accounting-convention markers for aggregate/total rows. Word-boundary
# matched so we don't drop legitimate rows but do catch "Total"/"Subtotal"/"Grand
# Total" rows that would otherwise double-count a sum or win a max.
_AGGREGATE_ROW_PATTERN = re.compile(r"\b(?:sub-?total|grand\s+total|total)\b", re.IGNORECASE)


@dataclass
class _Computed:
    operation: str
    value: float | int
    matched_row: dict | None
    excluded_aggregate_rows: int = 0


def _is_aggregate_row(row: dict) -> bool:
    """True if the row's label column (the first, in source column order) reads
    as an accounting total/subtotal marker. Only that column is checked —
    scanning every cell would misfire on an unrelated column that happens to
    contain the word "total" (e.g. a notes/commentary field), wrongly excluding
    a legitimate row from the aggregation."""
    if not row:
        return False
    label = next(iter(row.values()))
    return isinstance(label, str) and bool(_AGGREGATE_ROW_PATTERN.search(label))


def _tokens(text: str) -> set[str]:
    """Lowercased whole-word tokens — avoids 'summary'→'sum', 'account'→'count'."""
    return set(re.findall(r"[a-z]+", text.lower()))


def _try_compute(
    rows: list[dict],
    column: str | None,
    aggregation: str,
) -> _Computed | None:
    """
    Map a free-form aggregation description to a pandas operation.
    Returns None when no numeric operation is recognised — raw rows are used instead.
    """
    if not column or not aggregation:
        return None

    df = pd.DataFrame(rows)
    if column not in df.columns:
        return None

    tokens = _tokens(aggregation)

    # count works on any column type — check before the numeric guard.
    if "count" in tokens:
        return _Computed("count", int(df[column].count()), None)

    # Exclude accounting total/subtotal rows so sums don't double-count and
    # max/min don't return the "Total" row as the answer.
    agg_mask = df.apply(lambda r: _is_aggregate_row(r.to_dict()), axis=1)
    excluded = int(agg_mask.sum())
    data = df.loc[~agg_mask] if excluded else df

    numeric = pd.to_numeric(data[column], errors="coerce")
    if numeric.isna().all():
        return None

    if "sum" in tokens or "total" in tokens:
        return _Computed("sum", float(numeric.sum()), None, excluded)

    if tokens & {"max", "highest", "largest", "most", "greatest", "top", "maximum"}:
        idx = numeric.idxmax()
        return _Computed("max", float(numeric.max()), data.loc[idx].to_dict(), excluded)

    if tokens & {"min", "lowest", "smallest", "least", "fewest", "minimum"}:
        idx = numeric.idxmin()
        return _Computed("min", float(numeric.min()), data.loc[idx].to_dict(), excluded)

    if tokens & {"avg", "mean", "average"}:
        return _Computed("avg", float(numeric.mean()), None, excluded)

    return None  # unrecognised — let synthesizer reason over raw rows
