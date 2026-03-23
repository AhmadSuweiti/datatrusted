"""Integration tests for the top-level audit() function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datatrust import audit, Validator
from datatrust.report import AuditReport


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(99)
    n = 150
    return pd.DataFrame(
        {
            "user_id": range(1, n + 1),
            "age": rng.integers(18, 75, n).astype(float),
            "income": rng.normal(55_000, 20_000, n),
            "country": rng.choice(["US", "UK", "DE", "FR"], n),
            "signup_date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target": rng.choice([0, 1], n, p=[0.75, 0.25]),
        }
    )


class TestAuditFunction:
    def test_returns_audit_report(self, sample_df):
        report = audit(sample_df)
        assert isinstance(report, AuditReport)

    def test_shape_matches_input(self, sample_df):
        report = audit(sample_df)
        assert report.shape == sample_df.shape

    def test_all_sub_reports_populated(self, sample_df):
        report = audit(sample_df, target="target", id_columns=["user_id"])
        assert report.schema_report is not None
        assert report.missing_info is not None
        assert report.duplicate_info is not None
        assert report.target_info is not None

    def test_no_target_means_no_target_info(self, sample_df):
        report = audit(sample_df)
        assert report.target_info is None

    def test_id_columns_checked(self, sample_df):
        report = audit(sample_df, id_columns=["user_id"])
        assert "user_id" in report.duplicate_info.id_column_duplicates

    def test_validator_runs(self, sample_df):
        v = Validator().not_null("user_id").unique("user_id")
        report = audit(sample_df, validator=v)
        assert report.validation_result is not None
        assert report.validation_result.is_valid

    def test_type_error_on_non_dataframe(self):
        with pytest.raises(TypeError):
            audit({"a": [1, 2, 3]})

    def test_leakage_checks_can_be_disabled(self, sample_df):
        report = audit(sample_df, check_leakage=False)
        assert report.leakage_hints == []

    def test_missing_threshold_passed_through(self):
        df = pd.DataFrame({"x": [1, None, None, None, None]})  # 80 % missing
        report_strict = audit(df, missing_threshold=0.1)
        report_lenient = audit(df, missing_threshold=0.9)
        assert "x" in report_strict.missing_info.columns_above_threshold
        assert "x" not in report_lenient.missing_info.columns_above_threshold

    def test_outlier_infos_populated_for_skewed_data(self):
        rng = np.random.default_rng(3)
        vals = list(rng.normal(0, 1, 100)) + [1000, -1000]  # extreme outliers
        df = pd.DataFrame({"value": vals})
        report = audit(df)
        outlier_cols = [o.column for o in report.outlier_infos]
        assert "value" in outlier_cols

    def test_score_is_integer(self, sample_df):
        report = audit(sample_df)
        assert isinstance(report.score, int)

    def test_warnings_is_list_of_strings(self, sample_df):
        report = audit(sample_df)
        assert isinstance(report.warnings, list)
        for w in report.warnings:
            assert isinstance(w, str)

    def test_high_missing_lowers_score(self):
        df_clean = pd.DataFrame({"a": list(range(100))})
        nulls = [None] * 60 + list(range(40))
        df_miss = pd.DataFrame({"a": nulls})
        r_clean = audit(df_clean)
        r_miss = audit(df_miss)
        assert r_miss.score < r_clean.score
