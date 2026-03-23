"""Tests for datatrusted.rules — Validator and all built-in rules."""

from __future__ import annotations

import pandas as pd
import pytest

from datatrusted.rules import (
    AllowedValuesRule,
    DateNotInFutureRule,
    NonNegativeRule,
    NotNullRule,
    NumericRangeRule,
    UniqueRule,
    Validator,
)


# ---------------------------------------------------------------------------
# NotNullRule
# ---------------------------------------------------------------------------

class TestNotNullRule:
    def test_passes_on_clean_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert NotNullRule("x").validate(df) is None

    def test_fails_when_nulls_present(self):
        df = pd.DataFrame({"x": [1, None, 3]})
        v = NotNullRule("x").validate(df)
        assert v is not None
        assert v.affected_rows == 1
        assert v.rule == "not_null"

    def test_fails_on_missing_column(self):
        df = pd.DataFrame({"y": [1]})
        v = NotNullRule("x").validate(df)
        assert v is not None
        assert "does not exist" in v.description

    def test_sample_values_not_populated_for_nulls(self):
        df = pd.DataFrame({"x": [None, None]})
        v = NotNullRule("x").validate(df)
        # sample_values for all-null column will be empty
        assert v is not None


# ---------------------------------------------------------------------------
# UniqueRule
# ---------------------------------------------------------------------------

class TestUniqueRule:
    def test_passes_on_unique_column(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        assert UniqueRule("id").validate(df) is None

    def test_fails_on_duplicate_values(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        v = UniqueRule("id").validate(df)
        assert v is not None
        assert v.affected_rows == 2  # both occurrences of 2 are flagged
        assert v.rule == "unique"

    def test_sample_values_included(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3, 3]})
        v = UniqueRule("id").validate(df)
        assert v is not None
        assert len(v.sample_values) > 0


# ---------------------------------------------------------------------------
# NumericRangeRule
# ---------------------------------------------------------------------------

class TestNumericRangeRule:
    def test_passes_when_all_in_range(self):
        df = pd.DataFrame({"age": [18, 25, 40, 65]})
        assert NumericRangeRule("age", 0, 120).validate(df) is None

    def test_fails_on_values_below_min(self):
        df = pd.DataFrame({"age": [18, -1, 25]})
        v = NumericRangeRule("age", 0, 120).validate(df)
        assert v is not None
        assert v.affected_rows == 1

    def test_fails_on_values_above_max(self):
        df = pd.DataFrame({"age": [18, 25, 200]})
        v = NumericRangeRule("age", 0, 120).validate(df)
        assert v is not None
        assert v.affected_rows == 1

    def test_requires_at_least_one_bound(self):
        with pytest.raises(ValueError):
            NumericRangeRule("age")

    def test_min_only(self):
        df = pd.DataFrame({"score": [0.5, 1.2, -0.1]})
        v = NumericRangeRule("score", min_val=0.0).validate(df)
        assert v is not None
        assert v.affected_rows == 1

    def test_max_only(self):
        df = pd.DataFrame({"score": [0.5, 1.2, 0.1]})
        v = NumericRangeRule("score", max_val=1.0).validate(df)
        assert v is not None
        assert v.affected_rows == 1

    def test_nulls_are_skipped(self):
        df = pd.DataFrame({"age": [18, None, 25]})
        assert NumericRangeRule("age", 0, 120).validate(df) is None


# ---------------------------------------------------------------------------
# NonNegativeRule
# ---------------------------------------------------------------------------

class TestNonNegativeRule:
    def test_passes_on_non_negative_values(self):
        df = pd.DataFrame({"price": [0.0, 1.5, 100.0]})
        assert NonNegativeRule("price").validate(df) is None

    def test_fails_on_negative_values(self):
        df = pd.DataFrame({"price": [0.0, -1.5, 100.0]})
        v = NonNegativeRule("price").validate(df)
        assert v is not None
        assert v.affected_rows == 1
        assert v.rule == "non_negative"

    def test_zero_is_allowed(self):
        df = pd.DataFrame({"x": [0, 0, 0]})
        assert NonNegativeRule("x").validate(df) is None


# ---------------------------------------------------------------------------
# DateNotInFutureRule
# ---------------------------------------------------------------------------

class TestDateNotInFutureRule:
    def test_passes_on_past_dates(self):
        df = pd.DataFrame({"created_at": pd.to_datetime(["2020-01-01", "2021-06-15"])})
        assert DateNotInFutureRule("created_at").validate(df) is None

    def test_fails_on_future_dates(self):
        df = pd.DataFrame({"created_at": pd.to_datetime(["2020-01-01", "2099-01-01"])})
        v = DateNotInFutureRule("created_at").validate(df)
        assert v is not None
        assert v.affected_rows == 1
        assert v.rule == "date_not_in_future"

    def test_works_on_string_dates(self):
        df = pd.DataFrame({"dt": ["2020-01-01", "2099-12-31"]})
        v = DateNotInFutureRule("dt").validate(df)
        assert v is not None
        assert v.affected_rows == 1


# ---------------------------------------------------------------------------
# AllowedValuesRule
# ---------------------------------------------------------------------------

class TestAllowedValuesRule:
    def test_passes_when_all_allowed(self):
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        assert AllowedValuesRule("status", ["active", "inactive"]).validate(df) is None

    def test_fails_on_unexpected_value(self):
        df = pd.DataFrame({"status": ["active", "deleted", "active"]})
        v = AllowedValuesRule("status", ["active", "inactive"]).validate(df)
        assert v is not None
        assert v.affected_rows == 1

    def test_nulls_are_ignored(self):
        df = pd.DataFrame({"status": ["active", None, "inactive"]})
        assert AllowedValuesRule("status", ["active", "inactive"]).validate(df) is None


# ---------------------------------------------------------------------------
# Validator builder
# ---------------------------------------------------------------------------

class TestValidator:
    def test_empty_validator_always_valid(self, simple_df):
        result = Validator().validate(simple_df)
        assert result.is_valid
        assert result.violation_count == 0

    def test_chaining_returns_self(self):
        v = Validator()
        returned = v.not_null("x")
        assert returned is v

    def test_multiple_rules_all_pass(self):
        df = pd.DataFrame({"id": [1, 2, 3], "age": [20, 30, 40]})
        result = (
            Validator()
            .not_null("id")
            .unique("id")
            .in_range("age", 0, 120)
            .validate(df)
        )
        assert result.is_valid

    def test_collects_multiple_violations(self):
        df = pd.DataFrame({"id": [1, 1, None], "age": [-1, 200, 30]})
        result = (
            Validator()
            .not_null("id")
            .unique("id")
            .in_range("age", 0, 120)
            .validate(df)
        )
        assert not result.is_valid
        assert result.violation_count >= 2

    def test_violations_for_column(self):
        df = pd.DataFrame({"id": [1, 1, None]})
        result = (
            Validator()
            .not_null("id")
            .unique("id")
            .validate(df)
        )
        id_violations = result.violations_for("id")
        assert len(id_violations) == 2

    def test_rule_count(self):
        v = Validator().not_null("a").unique("b").non_negative("c")
        assert v.rule_count == 3

    def test_custom_rule_via_add_rule(self):
        from datatrusted.rules import NotNullRule
        df = pd.DataFrame({"x": [1, None]})
        result = Validator().add_rule(NotNullRule("x")).validate(df)
        assert not result.is_valid
