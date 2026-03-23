"""Tests for datatrust.duplicates."""

from __future__ import annotations

import pandas as pd
import pytest

from datatrust.duplicates import analyze_duplicates


class TestAnalyzeDuplicates:
    def test_no_duplicates(self, simple_df):
        info = analyze_duplicates(simple_df)
        assert info.full_row_duplicates == 0
        assert info.duplicate_pct == 0.0

    def test_detects_full_row_duplicates(self, df_with_duplicates):
        info = analyze_duplicates(df_with_duplicates)
        # Rows (2,20) and (4,40) each appear twice → 2 duplicate rows
        assert info.full_row_duplicates == 2
        assert info.duplicate_pct == pytest.approx(2 / 7)

    def test_total_rows_matches(self, df_with_duplicates):
        info = analyze_duplicates(df_with_duplicates)
        assert info.total_rows == len(df_with_duplicates)

    def test_id_column_duplicates(self, df_with_duplicates):
        info = analyze_duplicates(df_with_duplicates, id_columns=["id"])
        # id=2 and id=4 each appear twice → 2 extra rows
        assert info.id_column_duplicates["id"] == 2

    def test_missing_id_column_ignored(self, simple_df):
        info = analyze_duplicates(simple_df, id_columns=["nonexistent"])
        assert "nonexistent" not in info.id_column_duplicates

    def test_empty_dataframe(self):
        info = analyze_duplicates(pd.DataFrame({"a": []}))
        assert info.full_row_duplicates == 0
        assert info.total_rows == 0

    def test_all_identical_rows(self):
        df = pd.DataFrame({"x": [1, 1, 1], "y": [2, 2, 2]})
        info = analyze_duplicates(df)
        assert info.full_row_duplicates == 2  # first occurrence is not a duplicate
        assert info.duplicate_pct == pytest.approx(2 / 3)

    def test_unique_id_column_no_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4], "val": [10, 20, 30, 40]})
        info = analyze_duplicates(df, id_columns=["id"])
        assert info.id_column_duplicates["id"] == 0

    def test_multiple_id_columns(self):
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 3],
                "order_id": [100, 200, 100, 101],
                "amount": [50, 60, 70, 80],
            }
        )
        info = analyze_duplicates(df, id_columns=["user_id", "order_id"])
        # user_id=1 appears twice → 1 extra row
        assert info.id_column_duplicates["user_id"] == 1
        # order_id=100 appears twice → 1 extra row
        assert info.id_column_duplicates["order_id"] == 1
