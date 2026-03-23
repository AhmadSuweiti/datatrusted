"""
Rule-based validation system.

Provides a fluent ``Validator`` builder and a set of built-in rules.
Rules can be composed and run against any pandas DataFrame to produce
structured :class:`~datatrusted.models.ValidationResult` objects.

Example
-------
::

    from datatrusted import Validator

    result = (
        Validator()
        .not_null("email")
        .unique("user_id")
        .in_range("age", 0, 120)
        .non_negative("price")
        .date_not_in_future("signup_date")
        .validate(df)
    )

    if not result.is_valid:
        for v in result.violations:
            print(v.column, v.description)
"""

from __future__ import annotations

import datetime
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import pandas as pd

from datatrusted.models import RuleViolation, ValidationResult
from datatrusted.utils import safe_sample

Number = Union[int, float]


# ---------------------------------------------------------------------------
# Base rule
# ---------------------------------------------------------------------------


class Rule(ABC):
    """Abstract base class for all validation rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this rule, e.g. 'not_null'."""

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Optional[RuleViolation]:
        """Run the rule against *df* and return a violation, or None if clean."""


# ---------------------------------------------------------------------------
# Built-in rules
# ---------------------------------------------------------------------------


class NotNullRule(Rule):
    """Fails when any value in ``column`` is null / NaN."""

    def __init__(self, column: str) -> None:
        self.column = column

    @property
    def name(self) -> str:
        return "not_null"

    def validate(self, df: pd.DataFrame) -> Optional[RuleViolation]:
        if self.column not in df.columns:
            return RuleViolation(
                rule=self.name,
                column=self.column,
                description=f"Column '{self.column}' does not exist.",
                affected_rows=len(df),
            )
        null_mask = df[self.column].isnull()
        count = int(null_mask.sum())
        if count == 0:
            return None
        return RuleViolation(
            rule=self.name,
            column=self.column,
            description=f"Column '{self.column}' has {count} null value(s).",
            affected_rows=count,
            sample_values=safe_sample(df.loc[null_mask, self.column]),
        )


class UniqueRule(Rule):
    """Fails when any value in ``column`` appears more than once."""

    def __init__(self, column: str) -> None:
        self.column = column

    @property
    def name(self) -> str:
        return "unique"

    def validate(self, df: pd.DataFrame) -> Optional[RuleViolation]:
        if self.column not in df.columns:
            return RuleViolation(
                rule=self.name,
                column=self.column,
                description=f"Column '{self.column}' does not exist.",
                affected_rows=len(df),
            )
        duped_mask = df[self.column].duplicated(keep=False)
        count = int(duped_mask.sum())
        if count == 0:
            return None
        sample = safe_sample(df.loc[duped_mask, self.column])
        return RuleViolation(
            rule=self.name,
            column=self.column,
            description=(
                f"Column '{self.column}' has {count} row(s) with duplicate values."
            ),
            affected_rows=count,
            sample_values=sample,
        )


class NumericRangeRule(Rule):
    """Fails when values in ``column`` fall outside [min_val, max_val]."""

    def __init__(
        self,
        column: str,
        min_val: Optional[Number] = None,
        max_val: Optional[Number] = None,
    ) -> None:
        if min_val is None and max_val is None:
            raise ValueError("NumericRangeRule requires at least one of min_val or max_val.")
        self.column = column
        self.min_val = min_val
        self.max_val = max_val

    @property
    def name(self) -> str:
        return "in_range"

    def validate(self, df: pd.DataFrame) -> Optional[RuleViolation]:
        if self.column not in df.columns:
            return RuleViolation(
                rule=self.name,
                column=self.column,
                description=f"Column '{self.column}' does not exist.",
                affected_rows=len(df),
            )
        series = pd.to_numeric(df[self.column], errors="coerce").dropna()
        mask = pd.Series([True] * len(series), index=series.index)
        if self.min_val is not None:
            mask &= series < self.min_val
        if self.max_val is not None:
            mask |= series > self.max_val

        # Build the actual violation mask
        out_mask = pd.Series([False] * len(series), index=series.index)
        if self.min_val is not None:
            out_mask |= series < self.min_val
        if self.max_val is not None:
            out_mask |= series > self.max_val

        count = int(out_mask.sum())
        if count == 0:
            return None

        bounds = []
        if self.min_val is not None:
            bounds.append(f"min={self.min_val}")
        if self.max_val is not None:
            bounds.append(f"max={self.max_val}")
        return RuleViolation(
            rule=self.name,
            column=self.column,
            description=(
                f"Column '{self.column}' has {count} value(s) outside "
                f"the allowed range [{', '.join(bounds)}]."
            ),
            affected_rows=count,
            sample_values=safe_sample(series[out_mask]),
        )


class NonNegativeRule(Rule):
    """Fails when any value in ``column`` is strictly negative."""

    def __init__(self, column: str) -> None:
        self.column = column

    @property
    def name(self) -> str:
        return "non_negative"

    def validate(self, df: pd.DataFrame) -> Optional[RuleViolation]:
        if self.column not in df.columns:
            return RuleViolation(
                rule=self.name,
                column=self.column,
                description=f"Column '{self.column}' does not exist.",
                affected_rows=len(df),
            )
        series = pd.to_numeric(df[self.column], errors="coerce").dropna()
        neg_mask = series < 0
        count = int(neg_mask.sum())
        if count == 0:
            return None
        return RuleViolation(
            rule=self.name,
            column=self.column,
            description=f"Column '{self.column}' has {count} negative value(s).",
            affected_rows=count,
            sample_values=safe_sample(series[neg_mask]),
        )


class DateNotInFutureRule(Rule):
    """Fails when any date/datetime value in ``column`` is in the future.

    Works with columns that are already ``datetime64`` dtype or that can be
    parsed with ``pd.to_datetime``.
    """

    def __init__(self, column: str) -> None:
        self.column = column

    @property
    def name(self) -> str:
        return "date_not_in_future"

    def validate(self, df: pd.DataFrame) -> Optional[RuleViolation]:
        if self.column not in df.columns:
            return RuleViolation(
                rule=self.name,
                column=self.column,
                description=f"Column '{self.column}' does not exist.",
                affected_rows=len(df),
            )
        try:
            dates = pd.to_datetime(df[self.column], errors="coerce").dropna()
        except Exception:
            return RuleViolation(
                rule=self.name,
                column=self.column,
                description=f"Column '{self.column}' could not be parsed as dates.",
                affected_rows=0,
            )

        now = pd.Timestamp.now(tz=dates.dt.tz if hasattr(dates.dt, "tz") and dates.dt.tz else None)
        future_mask = dates > now
        count = int(future_mask.sum())
        if count == 0:
            return None
        return RuleViolation(
            rule=self.name,
            column=self.column,
            description=f"Column '{self.column}' has {count} date(s) in the future.",
            affected_rows=count,
            sample_values=[str(v) for v in safe_sample(dates[future_mask])],
        )


class AllowedValuesRule(Rule):
    """Fails when values in ``column`` are not in the allowed set."""

    def __init__(self, column: str, allowed: List[Any]) -> None:
        self.column = column
        self.allowed = set(allowed)

    @property
    def name(self) -> str:
        return "allowed_values"

    def validate(self, df: pd.DataFrame) -> Optional[RuleViolation]:
        if self.column not in df.columns:
            return RuleViolation(
                rule=self.name,
                column=self.column,
                description=f"Column '{self.column}' does not exist.",
                affected_rows=len(df),
            )
        non_null = df[self.column].dropna()
        invalid_mask = ~non_null.isin(self.allowed)
        count = int(invalid_mask.sum())
        if count == 0:
            return None
        listed = sorted(str(v) for v in list(self.allowed)[:10])
        return RuleViolation(
            rule=self.name,
            column=self.column,
            description=(
                f"Column '{self.column}' has {count} value(s) not in the "
                f"allowed set: {{{', '.join(listed)}{'...' if len(self.allowed) > 10 else ''}}}."
            ),
            affected_rows=count,
            sample_values=safe_sample(non_null[invalid_mask]),
        )


# ---------------------------------------------------------------------------
# Validator builder
# ---------------------------------------------------------------------------


class Validator:
    """Fluent builder for composing and running validation rules.

    Example
    -------
    ::

        result = (
            Validator()
            .not_null("user_id")
            .unique("user_id")
            .in_range("age", 0, 120)
            .validate(df)
        )
    """

    def __init__(self) -> None:
        self._rules: List[Rule] = []

    # ---- chainable rule-adders ----

    def add_rule(self, rule: Rule) -> "Validator":
        """Add any :class:`Rule` instance directly."""
        self._rules.append(rule)
        return self

    def not_null(self, column: str) -> "Validator":
        """Require no null values in *column*."""
        return self.add_rule(NotNullRule(column))

    def unique(self, column: str) -> "Validator":
        """Require all values in *column* to be unique."""
        return self.add_rule(UniqueRule(column))

    def in_range(
        self,
        column: str,
        min_val: Optional[Number] = None,
        max_val: Optional[Number] = None,
    ) -> "Validator":
        """Require numeric values in *column* to lie within [min_val, max_val]."""
        return self.add_rule(NumericRangeRule(column, min_val, max_val))

    def non_negative(self, column: str) -> "Validator":
        """Require all values in *column* to be ≥ 0."""
        return self.add_rule(NonNegativeRule(column))

    def date_not_in_future(self, column: str) -> "Validator":
        """Require all dates in *column* to not be in the future."""
        return self.add_rule(DateNotInFutureRule(column))

    def allowed_values(self, column: str, allowed: List[Any]) -> "Validator":
        """Require all values in *column* to belong to the *allowed* set."""
        return self.add_rule(AllowedValuesRule(column, allowed))

    # ---- execution ----

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Run all registered rules against *df*.

        Returns
        -------
        ValidationResult
            ``is_valid`` is True only when zero violations are found.
        """
        violations: List[RuleViolation] = []
        for rule in self._rules:
            violation = rule.validate(df)
            if violation is not None:
                violations.append(violation)
        return ValidationResult(is_valid=len(violations) == 0, violations=violations)

    @property
    def rule_count(self) -> int:
        """Number of rules currently registered."""
        return len(self._rules)
