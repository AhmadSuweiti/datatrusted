"""Tests for datatrusted.joins."""

from __future__ import annotations

import pandas as pd
import pytest

from datatrusted.joins import check_join


class TestCheckJoin:
    def test_clean_one_to_one_join(self):
        left = pd.DataFrame({"id": [1, 2, 3], "a": ["x", "y", "z"]})
        right = pd.DataFrame({"id": [1, 2, 3], "b": [10, 20, 30]})
        report = check_join(left, right, on="id")
        assert report.left_duplicates == 0
        assert report.right_duplicates == 0
        assert report.unmatched_left == 0
        assert report.unmatched_right == 0
        assert report.is_many_to_many is False
        assert report.warnings == []

    def test_detects_left_duplicates(self):
        left = pd.DataFrame({"id": [1, 1, 2], "a": ["x", "x2", "y"]})
        right = pd.DataFrame({"id": [1, 2], "b": [10, 20]})
        report = check_join(left, right, on="id")
        assert report.left_duplicates == 1
        assert any("left" in w.lower() for w in report.warnings)

    def test_detects_right_duplicates(self):
        left = pd.DataFrame({"id": [1, 2], "a": ["x", "y"]})
        right = pd.DataFrame({"id": [1, 1, 2], "b": [10, 11, 20]})
        report = check_join(left, right, on="id")
        assert report.right_duplicates == 1

    def test_detects_many_to_many(self):
        left = pd.DataFrame({"id": [1, 1, 2], "a": ["x", "x2", "y"]})
        right = pd.DataFrame({"id": [1, 1, 2], "b": [10, 11, 20]})
        report = check_join(left, right, on="id")
        assert report.is_many_to_many is True
        assert any("many-to-many" in w.lower() for w in report.warnings)

    def test_detects_unmatched_left_keys(self):
        left = pd.DataFrame({"id": [1, 2, 3], "a": ["x", "y", "z"]})
        right = pd.DataFrame({"id": [1, 2], "b": [10, 20]})
        report = check_join(left, right, on="id")
        assert report.unmatched_left == 1  # id=3 has no match

    def test_detects_unmatched_right_keys(self):
        left = pd.DataFrame({"id": [1, 2], "a": ["x", "y"]})
        right = pd.DataFrame({"id": [1, 2, 3], "b": [10, 20, 30]})
        report = check_join(left, right, on="id")
        assert report.unmatched_right == 1

    def test_missing_key_column_adds_warning(self):
        left = pd.DataFrame({"id": [1, 2]})
        right = pd.DataFrame({"other_id": [1, 2]})
        report = check_join(left, right, on="id")
        assert any("not found in right" in w for w in report.warnings)

    def test_composite_key(self):
        left = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 1], "val": [10, 20, 30]})
        right = pd.DataFrame({"a": [1, 1, 2], "b": [1, 2, 1], "info": ["x", "y", "z"]})
        report = check_join(left, right, on=["a", "b"])
        assert report.left_duplicates == 0
        assert report.right_duplicates == 0
        assert report.unmatched_left == 0

    def test_to_dict_keys(self):
        left = pd.DataFrame({"id": [1, 2]})
        right = pd.DataFrame({"id": [1, 2]})
        report = check_join(left, right, on="id")
        d = report.to_dict()
        assert "on" in d
        assert "left_duplicates" in d
        assert "is_many_to_many" in d
        assert "warnings" in d

    def test_shapes_recorded(self):
        left = pd.DataFrame({"id": [1, 2, 3]})
        right = pd.DataFrame({"id": [1, 2], "extra": ["a", "b"]})
        report = check_join(left, right, on="id")
        assert report.left_shape == (3, 1)
        assert report.right_shape == (2, 2)

    def test_string_key_normalised_to_list(self):
        left = pd.DataFrame({"id": [1]})
        right = pd.DataFrame({"id": [1]})
        report = check_join(left, right, on="id")
        assert isinstance(report.on, list)
        assert report.on == ["id"]
