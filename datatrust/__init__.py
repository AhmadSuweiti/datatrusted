"""
datatrust — A trust layer for tabular data.

Quick start::

    import pandas as pd
    from datatrust import audit, compare_splits, check_join, Validator

    df = pd.read_csv("data.csv")

    # Full dataset audit
    report = audit(df, target="label", id_columns=["id"])
    print(report.score)
    report.to_html("report.html")

    # Custom validation rules
    result = (
        Validator()
        .not_null("email")
        .unique("user_id")
        .in_range("age", 0, 120)
        .validate(df)
    )

    # Train/test drift
    drift = compare_splits(train_df, test_df)

    # Join integrity
    join_report = check_join(orders, customers, on="customer_id")
"""

from datatrust.audit import audit
from datatrust.drift import compare_splits
from datatrust.joins import check_join
from datatrust.models import (
    DriftReport,
    DuplicateInfo,
    JoinReport,
    LeakageHint,
    MissingInfo,
    OutlierInfo,
    RuleViolation,
    SchemaIssue,
    SchemaReport,
    TargetInfo,
    ValidationResult,
)
from datatrust.report import AuditReport
from datatrust.rules import (
    AllowedValuesRule,
    DateNotInFutureRule,
    NonNegativeRule,
    NotNullRule,
    NumericRangeRule,
    Rule,
    UniqueRule,
    Validator,
)

__version__ = "0.1.0"
__author__ = "datatrust contributors"

__all__ = [
    # Main API
    "audit",
    "compare_splits",
    "check_join",
    "Validator",
    # Report objects
    "AuditReport",
    "DriftReport",
    "JoinReport",
    "ValidationResult",
    # Data models
    "RuleViolation",
    "SchemaIssue",
    "SchemaReport",
    "MissingInfo",
    "DuplicateInfo",
    "OutlierInfo",
    "TargetInfo",
    "LeakageHint",
    # Rule classes (for custom rules)
    "Rule",
    "NotNullRule",
    "UniqueRule",
    "NumericRangeRule",
    "NonNegativeRule",
    "DateNotInFutureRule",
    "AllowedValuesRule",
    # Version
    "__version__",
]
