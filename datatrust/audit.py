"""
Main audit entrypoint.

:func:`audit` is the primary public function of datatrust. It orchestrates
all checks and returns a single :class:`~datatrust.report.AuditReport`.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from datatrust.duplicates import analyze_duplicates
from datatrust.leakage import analyze_leakage
from datatrust.missing import analyze_missing
from datatrust.models import ValidationResult
from datatrust.outliers import analyze_outliers
from datatrust.report import AuditReport
from datatrust.rules import Validator
from datatrust.schema import analyze_schema
from datatrust.target import analyze_target


def audit(
    df: pd.DataFrame,
    *,
    target: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    datetime_columns: Optional[List[str]] = None,
    missing_threshold: float = 0.05,
    validator: Optional[Validator] = None,
    check_leakage: bool = True,
) -> AuditReport:
    """Run a full audit on a tabular DataFrame.

    This is the main entrypoint for datatrust. It runs schema checks,
    missing-value analysis, duplicate detection, outlier detection, target
    analysis, and optional leakage hints, then returns a single
    :class:`~datatrust.report.AuditReport`.

    Parameters
    ----------
    df:
        The DataFrame to audit.
    target:
        Name of the target/label column (optional). When provided, extra
        checks are run on class distribution, missing labels, and leakage.
    id_columns:
        Columns that should be unique identifiers (e.g. ``['user_id']``).
        Duplicate values in these columns are reported separately.
    datetime_columns:
        Column names the caller knows are datetime but may not be parsed.
        These are passed as hints to the schema checker.
    missing_threshold:
        Fraction (0–1). Columns with more than this fraction of missing
        values are flagged. Default is ``0.05`` (5 %).
    validator:
        An optional pre-configured :class:`~datatrust.rules.Validator` to
        run against the DataFrame. Build one with the fluent API::

            from datatrust import Validator
            v = Validator().not_null("email").unique("user_id")
            report = audit(df, validator=v)

    check_leakage:
        Whether to run heuristic leakage checks. Default ``True``.

    Returns
    -------
    AuditReport
        A structured result with a trust score, warnings, and per-check
        sub-reports.

    Example
    -------
    ::

        import pandas as pd
        from datatrust import audit

        df = pd.read_csv("customers.csv")
        report = audit(df, target="churn", id_columns=["customer_id"])

        print(report.score)       # e.g. 74
        print(report.summary)     # plain-text summary
        report.to_html("audit.html")
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    # --- schema / type checks ---
    schema_report = analyze_schema(df, datetime_column_hints=datetime_columns)

    # --- missing values ---
    missing_info = analyze_missing(df, threshold=missing_threshold)

    # --- duplicates ---
    duplicate_info = analyze_duplicates(df, id_columns=id_columns)

    # --- outliers ---
    outlier_infos = analyze_outliers(df)

    # --- target analysis ---
    target_info = analyze_target(df, target) if target else None

    # --- leakage hints ---
    leakage_hints = []
    if check_leakage:
        leakage_hints = analyze_leakage(df, target_column=target)

    # --- custom validation ---
    validation_result: Optional[ValidationResult] = None
    if validator is not None:
        validation_result = validator.validate(df)

    return AuditReport(
        shape=df.shape,
        schema_report=schema_report,
        missing_info=missing_info,
        duplicate_info=duplicate_info,
        outlier_infos=outlier_infos,
        target_info=target_info,
        leakage_hints=leakage_hints,
        validation_result=validation_result,
    )
