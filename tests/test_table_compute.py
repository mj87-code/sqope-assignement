"""
Unit tests for the keyword-guided pandas compute used by the hybrid pipeline.
(The analytical pipeline now uses text-to-SQL — see test_analytical_pipeline.py.)
"""
SAMPLE_HEADCOUNT_ROWS = [
    {"Department": "R&D",          "Headcount Q3": 2450.0, "Headcount Q4": 2620.0},
    {"Department": "Sales",        "Headcount Q3": 1850.0, "Headcount Q4": 1920.0},
    {"Department": "Operations",   "Headcount Q3":  950.0, "Headcount Q4":  980.0},
    {"Department": "Engineering",  "Headcount Q3": 2100.0, "Headcount Q4": 2200.0},
    {"Department": "HR",           "Headcount Q3":  340.0, "Headcount Q4":  350.0},
    {"Department": "Finance",      "Headcount Q3":  220.0, "Headcount Q4":  230.0},
    {"Department": "Legal",        "Headcount Q3":  290.0, "Headcount Q4":  310.0},
]  # Q3 sum = 8200, Q4 max = R&D 2620

INCOME_ROWS = [
    {"Category": "Revenue - Cloud Services", "Q4 2024 (USD M)": 680.0},
    {"Category": "Revenue - AI Solutions",   "Q4 2024 (USD M)": 310.0},
    {"Category": "Revenue - Hardware",       "Q4 2024 (USD M)": 250.0},
    {"Category": "Total Revenue",            "Q4 2024 (USD M)": 1420.0},
    {"Category": "Net Income",               "Q4 2024 (USD M)": 230.0},
]


class TestTryCompute:
    def test_sum_recognised(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "sum all values")
        assert result.operation == "sum"
        assert result.value == 8200.0

    def test_total_keyword_maps_to_sum(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "find the total")
        assert result.operation == "sum"

    def test_max_with_matched_row(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q4", "find the row with the highest value")
        assert result.operation == "max"
        assert result.value == 2620.0
        assert result.matched_row["Department"] == "R&D"

    def test_min_with_matched_row(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "find the row with the smallest value")
        assert result.operation == "min"
        assert result.value == 220.0

    def test_avg_recognised(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "calculate the average")
        assert result.operation == "avg"

    def test_count_recognised(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Department", "count the number of rows")
        assert result.operation == "count"
        assert result.value == 7

    def test_compare_returns_none(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "compare values across all rows")
        assert result is None

    def test_missing_column_returns_none(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "NonExistentColumn", "sum all values")
        assert result is None

    def test_no_column_returns_none(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, None, "sum all values")
        assert result is None

    def test_non_numeric_column_returns_none(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Department", "sum all values")
        assert result is None


class TestAggregateRowExclusion:
    def test_sum_excludes_total_row(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(INCOME_ROWS, "Q4 2024 (USD M)", "sum all values")
        # Line items + Net Income (no 'total' label) = 680+310+250+230 = 1470.
        assert result.value == 1470.0
        assert result.excluded_aggregate_rows == 1

    def test_max_does_not_return_total_row(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(INCOME_ROWS, "Q4 2024 (USD M)", "row with the highest value")
        assert result.matched_row["Category"] != "Total Revenue"
        assert result.value == 680.0

    def test_no_exclusion_when_no_total_rows(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "sum all values")
        assert result.value == 8200.0
        assert result.excluded_aggregate_rows == 0


class TestTokenMatching:
    def test_summary_does_not_trigger_sum(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "provide a summary of the rows")
        assert result is None

    def test_account_does_not_trigger_count(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "the account balance per row")
        assert result is None

    def test_whole_word_sum_still_matches(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(SAMPLE_HEADCOUNT_ROWS, "Headcount Q3", "sum the column")
        assert result.operation == "sum"


class TestCategoryScopingGap:
    """KNOWN LIMITATION, currently failing on purpose (see FINDINGS.md #1's
    "residual limitation" note).

    _is_aggregate_row only excludes rows whose label literally contains
    "total"/"subtotal"/"grand total" — it has no concept of semantic category.
    On a statement table that mixes metric kinds in one column (revenue line
    items alongside a non-revenue metric like Net Income, which isn't labelled
    "total"), asking to sum "the revenue line items only" silently includes
    Net Income, because nothing about it trips the total/subtotal check.

    There is currently no fix available at the row_filter layer either:
    row_filter is exact JSONB containment (row_data @> {"Category": "X"}), so
    it can select one row by an exact value but cannot express "every row
    whose Category contains 'Revenue'" as a pattern across several rows.

    This test encodes the CORRECT answer a due-diligence tool should give and
    is expected to fail until category-aware scoping exists (e.g. row_filter
    supporting partial/pattern matches, or an LLM-driven category exclusion
    list threaded through _try_compute). Do not "fix" it by loosening the
    assertion — fix the underlying scoping gap, or explicitly mark it xfail
    with a tracking note if it's deliberately deferred.
    """

    def test_sum_of_revenue_lines_only_excludes_non_revenue_metrics(self):
        from pipeline.table_compute import _try_compute
        result = _try_compute(INCOME_ROWS, "Q4 2024 (USD M)", "sum the revenue line items only")
        # Correct answer: Cloud Services (680) + AI Solutions (310) + Hardware
        # (250) = 1240. Net Income (230) is a different metric, not a revenue
        # line item, and must NOT be included just because it isn't labelled
        # "total". Today's implementation returns 1470 (it also sums Net
        # Income), so this assertion currently fails.
        assert result.value == 1240.0
