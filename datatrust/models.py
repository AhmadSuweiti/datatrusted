"""
Dataclass models for all datatrust result objects.

These are the structured return types for every audit, check, and report
in the library. All are plain dataclasses — no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Validation models
# ---------------------------------------------------------------------------


@dataclass
class RuleViolation:
    """A single rule violation found during validation."""

    rule: str
    """Name of the rule that was violated."""

    column: str
    """Column where the violation was detected."""

    description: str
    """Human-readable description of the violation."""

    affected_rows: int
    """Number of rows that violate the rule."""

    sample_values: List[Any] = field(default_factory=list)
    """Up to 5 example values that triggered the violation."""


@dataclass
class ValidationResult:
    """Aggregated result of running a Validator over a DataFrame."""

    is_valid: bool
    """True if no violations were found."""

    violations: List[RuleViolation] = field(default_factory=list)

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def violations_for(self, column: str) -> List[RuleViolation]:
        """Return all violations for a specific column."""
        return [v for v in self.violations if v.column == column]


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------


@dataclass
class SchemaIssue:
    """A detected schema or type problem in a column."""

    column: str
    issue_type: str
    """One of: 'numeric_as_string', 'datetime_as_string', 'constant_column',
    'high_cardinality_object'."""

    description: str
    suggestion: str = ""


@dataclass
class SchemaReport:
    """Full results of schema/type analysis."""

    column_dtypes: Dict[str, str] = field(default_factory=dict)
    """Maps column name → pandas dtype string."""

    issues: List[SchemaIssue] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        return len(self.issues)


# ---------------------------------------------------------------------------
# Missing value models
# ---------------------------------------------------------------------------


@dataclass
class MissingInfo:
    """Missing value statistics for a DataFrame."""

    total_missing: int
    total_cells: int
    missing_by_column: Dict[str, int] = field(default_factory=dict)
    missing_pct_by_column: Dict[str, float] = field(default_factory=dict)
    columns_above_threshold: List[str] = field(default_factory=list)
    """Columns whose missing percentage exceeds the configured threshold."""
    threshold: float = 0.05

    @property
    def overall_missing_pct(self) -> float:
        if self.total_cells == 0:
            return 0.0
        return self.total_missing / self.total_cells


# ---------------------------------------------------------------------------
# Duplicate models
# ---------------------------------------------------------------------------


@dataclass
class DuplicateInfo:
    """Duplicate row statistics."""

    full_row_duplicates: int
    """Number of rows that are exact copies of another row."""

    total_rows: int
    duplicate_pct: float

    id_column_duplicates: Dict[str, int] = field(default_factory=dict)
    """Maps id column name → count of duplicate values in that column."""


# ---------------------------------------------------------------------------
# Outlier models
# ---------------------------------------------------------------------------


@dataclass
class OutlierInfo:
    """IQR-based outlier statistics for a single numeric column."""

    column: str
    outlier_count: int
    outlier_pct: float
    lower_bound: float
    """Values below this bound are considered outliers (Q1 - 1.5*IQR)."""
    upper_bound: float
    """Values above this bound are considered outliers (Q3 + 1.5*IQR)."""


# ---------------------------------------------------------------------------
# Target models
# ---------------------------------------------------------------------------


@dataclass
class TargetInfo:
    """Analysis of the designated target/label column."""

    column: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_likely_classification: bool
    """True when the column has low cardinality (≤ 20 unique values)."""

    class_counts: Dict[str, int] = field(default_factory=dict)
    """Maps class label → row count. Only populated for classification targets."""

    imbalance_ratio: Optional[float] = None
    """Ratio of majority class to minority class. None for regression targets."""

    cardinality_warning: bool = False
    """True when unique_count is very high (possible regression mislabelled as
    classification, or an ID-like column used as target)."""


# ---------------------------------------------------------------------------
# Leakage models
# ---------------------------------------------------------------------------


@dataclass
class LeakageHint:
    """A heuristic hint about potential data leakage.

    These are warnings only — not definitive detections. Always review manually.
    """

    column: str
    hint_type: str
    """One of: 'suspicious_name', 'high_target_correlation', 'near_constant_after_split'."""

    description: str
    severity: str = "medium"
    """'low' | 'medium' | 'high'."""


# ---------------------------------------------------------------------------
# Drift models
# ---------------------------------------------------------------------------


@dataclass
class DriftInfo:
    """Distribution drift information for a single column."""

    column: str
    column_type: str
    """'numeric' or 'categorical'."""

    drift_detected: bool
    drift_score: float
    """Normalized score in [0, 1]. Higher means more drift."""

    description: str

    # Categorical-only fields
    missing_in_test: List[str] = field(default_factory=list)
    """Categories present in train but absent in test."""
    unseen_in_test: List[str] = field(default_factory=list)
    """Categories present in test but absent in train."""


@dataclass
class DriftReport:
    """Results of train/test distribution comparison."""

    train_shape: Tuple[int, int] = field(default=(0, 0))
    test_shape: Tuple[int, int] = field(default=(0, 0))
    numeric_drifts: List[DriftInfo] = field(default_factory=list)
    categorical_drifts: List[DriftInfo] = field(default_factory=list)

    @property
    def drifted_columns(self) -> List[str]:
        """All columns where drift was detected."""
        return [
            d.column
            for d in self.numeric_drifts + self.categorical_drifts
            if d.drift_detected
        ]

    @property
    def drift_count(self) -> int:
        return len(self.drifted_columns)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_shape": self.train_shape,
            "test_shape": self.test_shape,
            "drifted_columns": self.drifted_columns,
            "drift_count": self.drift_count,
            "numeric_drifts": [
                {
                    "column": d.column,
                    "drift_detected": d.drift_detected,
                    "drift_score": round(d.drift_score, 4),
                    "description": d.description,
                }
                for d in self.numeric_drifts
            ],
            "categorical_drifts": [
                {
                    "column": d.column,
                    "drift_detected": d.drift_detected,
                    "drift_score": round(d.drift_score, 4),
                    "description": d.description,
                    "missing_in_test": d.missing_in_test,
                    "unseen_in_test": d.unseen_in_test,
                }
                for d in self.categorical_drifts
            ],
        }


# ---------------------------------------------------------------------------
# Join models
# ---------------------------------------------------------------------------


@dataclass
class JoinReport:
    """Results of join integrity checks between two DataFrames."""

    on: List[str]
    """Join key column names."""

    left_shape: Tuple[int, int]
    right_shape: Tuple[int, int]
    left_duplicates: int
    """Rows in the left frame with duplicate join key values."""
    right_duplicates: int
    """Rows in the right frame with duplicate join key values."""
    unmatched_left: int
    """Keys present in left but absent in right."""
    unmatched_right: int
    """Keys present in right but absent in left."""
    is_many_to_many: bool
    """True when both sides have duplicate keys — join will fan out rows."""

    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "on": self.on,
            "left_shape": self.left_shape,
            "right_shape": self.right_shape,
            "left_duplicates": self.left_duplicates,
            "right_duplicates": self.right_duplicates,
            "unmatched_left": self.unmatched_left,
            "unmatched_right": self.unmatched_right,
            "is_many_to_many": self.is_many_to_many,
            "warnings": self.warnings,
        }
