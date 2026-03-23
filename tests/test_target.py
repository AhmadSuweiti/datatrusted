"""Tests for datatrust.target."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datatrust.target import analyze_target


class TestAnalyzeTarget:
    def test_returns_none_for_missing_column(self, simple_df):
        result = analyze_target(simple_df, "nonexistent")
        assert result is None

    def test_classification_detected(self, classification_df):
        info = analyze_target(classification_df, "label")
        assert info is not None
        assert info.is_likely_classification is True
        assert info.column == "label"

    def test_missing_labels_counted(self):
        df = pd.DataFrame({"label": [0, 1, None, 1, 0, None]})
        info = analyze_target(df, "label")
        assert info.missing_count == 2
        assert abs(info.missing_pct - 2 / 6) < 1e-9

    def test_no_missing_labels(self, classification_df):
        info = analyze_target(classification_df, "label")
        assert info.missing_count == 0
        assert info.missing_pct == 0.0

    def test_class_counts_populated(self, classification_df):
        info = analyze_target(classification_df, "label")
        assert "0" in info.class_counts or 0 in info.class_counts
        total = sum(info.class_counts.values())
        assert total == len(classification_df)

    def test_imbalance_ratio_computed(self, classification_df):
        info = analyze_target(classification_df, "label")
        # 80/20 split → ratio should be around 4x
        assert info.imbalance_ratio is not None
        assert info.imbalance_ratio > 1.0

    def test_high_cardinality_warning(self):
        # Continuous target — many unique values
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"price": rng.uniform(0, 1000, 200)})
        info = analyze_target(df, "price")
        assert info is not None
        assert info.cardinality_warning is True
        assert info.is_likely_classification is False

    def test_binary_target_class_counts(self):
        df = pd.DataFrame({"y": [0, 0, 0, 1, 1]})
        info = analyze_target(df, "y")
        assert info is not None
        counts = list(info.class_counts.values())
        assert sorted(counts, reverse=True) == [3, 2]

    def test_unique_count_correct(self):
        df = pd.DataFrame({"category": ["a", "b", "c", "a", "b"]})
        info = analyze_target(df, "category")
        assert info.unique_count == 3
