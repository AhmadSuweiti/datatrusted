"""Tests for datatrusted.drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datatrusted.drift import compare_splits


def _make_train_test(n_train=500, n_test=200, seed=42):
    """Clean train/test split with no drift."""
    rng = np.random.default_rng(seed)
    train = pd.DataFrame(
        {
            "age": rng.normal(35, 10, n_train),
            "income": rng.normal(60_000, 15_000, n_train),
            "city": rng.choice(["London", "Paris", "Berlin"], n_train),
            "label": rng.choice([0, 1], n_train),
        }
    )
    test = pd.DataFrame(
        {
            "age": rng.normal(35, 10, n_test),
            "income": rng.normal(60_000, 15_000, n_test),
            "city": rng.choice(["London", "Paris", "Berlin"], n_test),
            "label": rng.choice([0, 1], n_test),
        }
    )
    return train, test


class TestCompareSplits:
    def test_returns_drift_report(self):
        train, test = _make_train_test()
        report = compare_splits(train, test)
        from datatrusted.models import DriftReport
        assert isinstance(report, DriftReport)

    def test_shapes_recorded(self):
        train, test = _make_train_test(500, 200)
        report = compare_splits(train, test)
        assert report.train_shape == (500, 4)
        assert report.test_shape == (200, 4)

    def test_no_drift_on_same_distribution(self):
        train, test = _make_train_test()
        report = compare_splits(train, test, target="label")
        # With same distributions and reasonable sample size there should be
        # very few or no drifted columns
        assert report.drift_count <= 1  # allow for random variation

    def test_detects_numeric_drift(self):
        rng = np.random.default_rng(0)
        train = pd.DataFrame({"x": rng.normal(0, 1, 500)})
        # Test mean is shifted by 3 standard deviations — definitely drift
        test = pd.DataFrame({"x": rng.normal(5, 1, 200)})
        report = compare_splits(train, test)
        assert len(report.numeric_drifts) == 1
        assert report.numeric_drifts[0].drift_detected is True
        assert "x" in report.drifted_columns

    def test_detects_categorical_drift(self):
        rng = np.random.default_rng(1)
        train = pd.DataFrame(
            {"cat": rng.choice(["a", "b", "c"], 300, p=[0.33, 0.33, 0.34])}
        )
        # Test has completely different distribution
        test = pd.DataFrame(
            {"cat": rng.choice(["a", "b", "c"], 100, p=[0.9, 0.05, 0.05])}
        )
        report = compare_splits(train, test)
        assert len(report.categorical_drifts) == 1
        assert report.categorical_drifts[0].drift_detected is True

    def test_missing_categories_detected(self):
        train = pd.DataFrame({"status": ["A", "B", "C", "A", "B"]})
        test = pd.DataFrame({"status": ["A", "A", "B"]})  # C is missing
        report = compare_splits(train, test)
        assert len(report.categorical_drifts) == 1
        drift = report.categorical_drifts[0]
        assert "C" in drift.missing_in_test

    def test_unseen_categories_detected(self):
        train = pd.DataFrame({"status": ["A", "B", "A", "B"]})
        test = pd.DataFrame({"status": ["A", "B", "D"]})  # D is new
        report = compare_splits(train, test)
        assert len(report.categorical_drifts) == 1
        drift = report.categorical_drifts[0]
        assert "D" in drift.unseen_in_test

    def test_target_excluded_from_drift(self):
        train, test = _make_train_test()
        report = compare_splits(train, test, target="label")
        checked_cols = [d.column for d in report.numeric_drifts + report.categorical_drifts]
        assert "label" not in checked_cols

    def test_drift_count_property(self):
        rng = np.random.default_rng(5)
        train = pd.DataFrame({"x": rng.normal(0, 1, 200)})
        test = pd.DataFrame({"x": rng.normal(10, 1, 200)})
        report = compare_splits(train, test)
        assert report.drift_count == len(report.drifted_columns)

    def test_to_dict_keys(self):
        train, test = _make_train_test()
        report = compare_splits(train, test)
        d = report.to_dict()
        assert "train_shape" in d
        assert "drifted_columns" in d
        assert "numeric_drifts" in d
        assert "categorical_drifts" in d
