"""Tests for datatrust.report — AuditReport, scoring, and export."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from datatrust import audit


def _make_clean_df(n=100):
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "age": rng.integers(18, 80, n),
            "score": rng.uniform(0, 100, n),
            "category": rng.choice(["A", "B", "C"], n),
        }
    )


def _make_dirty_df(n=100):
    rng = np.random.default_rng(8)
    df = pd.DataFrame(
        {
            "id": list(range(1, n + 1 - 5)) + [1, 2, 3, 4, 5],  # last 5 are dupes
            "age": [None if i % 5 == 0 else int(rng.integers(18, 80)) for i in range(n)],
            "score": rng.uniform(-10, 110, n),  # some negatives, some > 100
            "category": rng.choice(["A", "B", "C"], n),
            "label": rng.choice([0, 1], n, p=[0.9, 0.1]),
        }
    )
    return df


class TestAuditReport:
    def test_score_on_clean_data_is_high(self):
        df = _make_clean_df()
        report = audit(df)
        assert report.score >= 85

    def test_score_on_dirty_data_is_lower(self):
        clean = audit(_make_clean_df())
        dirty = audit(_make_dirty_df(), target="label")
        assert dirty.score < clean.score

    def test_score_bounded_0_to_100(self):
        for _ in range(5):
            df = _make_dirty_df()
            report = audit(df)
            assert 0 <= report.score <= 100

    def test_summary_is_string(self):
        df = _make_clean_df()
        report = audit(df)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 20

    def test_warnings_is_list(self):
        df = _make_clean_df()
        report = audit(df)
        assert isinstance(report.warnings, list)

    def test_to_dict_has_required_keys(self):
        df = _make_clean_df()
        report = audit(df)
        d = report.to_dict()
        assert "score" in d
        assert "shape" in d
        assert "warnings" in d

    def test_to_dict_json_serialisable(self):
        df = _make_dirty_df()
        report = audit(df, target="label")
        d = report.to_dict()
        # Should not raise
        json.dumps(d)

    def test_to_markdown_returns_string(self):
        df = _make_clean_df()
        report = audit(df)
        md = report.to_markdown()
        assert isinstance(md, str)
        assert "# datatrust" in md
        assert "Trust Score" in md

    def test_to_markdown_writes_file(self):
        df = _make_clean_df()
        report = audit(df)
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            report.to_markdown(path=path)
            assert os.path.exists(path)
            content = open(path).read()
            assert "Trust Score" in content
        finally:
            os.unlink(path)

    def test_to_html_returns_string(self):
        df = _make_clean_df()
        report = audit(df)
        html = report.to_html()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "datatrust" in html

    def test_to_html_writes_file(self):
        df = _make_clean_df()
        report = audit(df)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.to_html(path=path)
            assert os.path.exists(path)
            content = open(path).read()
            assert "<table" in content
        finally:
            os.unlink(path)

    def test_shape_stored_correctly(self):
        df = _make_clean_df(50)
        report = audit(df)
        assert report.shape == (50, 4)

    def test_missing_info_present(self):
        df = _make_dirty_df()
        report = audit(df)
        assert report.missing_info is not None

    def test_duplicate_info_present(self):
        df = _make_dirty_df()
        report = audit(df)
        assert report.duplicate_info is not None

    def test_target_info_none_when_not_specified(self):
        df = _make_clean_df()
        report = audit(df)
        assert report.target_info is None

    def test_target_info_populated_when_specified(self):
        df = _make_dirty_df()
        report = audit(df, target="label")
        assert report.target_info is not None
        assert report.target_info.column == "label"

    def test_validation_result_none_without_validator(self):
        report = audit(_make_clean_df())
        assert report.validation_result is None

    def test_validation_result_populated_with_validator(self):
        from datatrust import Validator
        df = _make_clean_df()
        v = Validator().not_null("id").unique("id")
        report = audit(df, validator=v)
        assert report.validation_result is not None
        assert report.validation_result.is_valid


class TestScoring:
    def test_penalty_for_high_missingness(self):
        df_clean = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        # 80 % of column a is missing
        df_miss = pd.DataFrame({"a": [None, None, None, None, 1]})
        r_clean = audit(df_clean)
        r_miss = audit(df_miss, check_leakage=False)
        assert r_miss.score < r_clean.score

    def test_penalty_for_duplicates(self):
        df_clean = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
        df_dup = pd.DataFrame({"id": [1, 1, 1, 1, 1]})  # all dupes
        r_clean = audit(df_clean)
        r_dup = audit(df_dup)
        assert r_dup.score < r_clean.score

    def test_invalid_input_raises_type_error(self):
        with pytest.raises(TypeError):
            audit([1, 2, 3])
