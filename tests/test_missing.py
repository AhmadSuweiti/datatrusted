"""Tests for datatrust.missing."""

from __future__ import annotations

import pandas as pd
import pytest

from datatrust.missing import analyze_missing, missing_summary_table


class TestAnalyzeMissing:
    def test_no_missing(self, simple_df):
        info = analyze_missing(simple_df)
        assert info.total_missing == 0
        assert info.columns_above_threshold == []
        assert info.overall_missing_pct == 0.0

    def test_counts_missing_correctly(self, df_with_missing):
        info = analyze_missing(df_with_missing)
        # age has 7 missing out of 10
        assert info.missing_by_column["age"] == 7
        assert abs(info.missing_pct_by_column["age"] - 0.7) < 1e-9

    def test_columns_above_threshold(self, df_with_missing):
        info = analyze_missing(df_with_missing, threshold=0.05)
        # age (70 %), salary (20 %), city (30 %) all exceed 5 %
        assert "age" in info.columns_above_threshold
        assert "salary" in info.columns_above_threshold
        assert "city" in info.columns_above_threshold

    def test_threshold_filters_correctly(self, df_with_missing):
        info = analyze_missing(df_with_missing, threshold=0.50)
        # Only age (70 %) should be above 50 %
        assert info.columns_above_threshold == ["age"]

    def test_total_missing_is_sum(self, df_with_missing):
        info = analyze_missing(df_with_missing)
        expected = sum(info.missing_by_column.values())
        assert info.total_missing == expected

    def test_overall_missing_pct(self, df_with_missing):
        info = analyze_missing(df_with_missing)
        expected = info.total_missing / info.total_cells
        assert abs(info.overall_missing_pct - expected) < 1e-9

    def test_empty_dataframe(self):
        info = analyze_missing(pd.DataFrame())
        assert info.total_missing == 0
        assert info.total_cells == 0
        assert info.overall_missing_pct == 0.0

    def test_all_missing_column(self):
        df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
        info = analyze_missing(df, threshold=0.5)
        assert info.missing_by_column["a"] == 3
        assert "a" in info.columns_above_threshold
        assert "b" not in info.columns_above_threshold

    def test_threshold_stored_in_result(self):
        df = pd.DataFrame({"x": [1, None, 3]})
        info = analyze_missing(df, threshold=0.20)
        assert info.threshold == 0.20


class TestMissingSummaryTable:
    def test_returns_dataframe(self, df_with_missing):
        info = analyze_missing(df_with_missing)
        table = missing_summary_table(info)
        assert isinstance(table, pd.DataFrame)

    def test_sorted_descending(self, df_with_missing):
        info = analyze_missing(df_with_missing)
        table = missing_summary_table(info)
        pcts = table["missing_pct"].tolist()
        assert pcts == sorted(pcts, reverse=True)

    def test_above_threshold_column(self, df_with_missing):
        info = analyze_missing(df_with_missing, threshold=0.50)
        table = missing_summary_table(info)
        above = table.loc[table["above_threshold"], "column"].tolist()
        assert "age" in above
