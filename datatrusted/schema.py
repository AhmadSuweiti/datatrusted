"""
Schema and type analysis.

Detects columns whose inferred dtype does not match their content —
e.g. numeric data stored as strings, unparsed datetime columns, or
columns that carry no information (constants).
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from datatrusted.models import SchemaIssue, SchemaReport
from datatrusted.utils import (
    categorical_columns,
    is_likely_datetime_string,
    is_likely_numeric_string,
    numeric_columns,
)

# Threshold: if a column's unique-value ratio exceeds this and the dtype is
# object, warn about potential high-cardinality noise.
_HIGH_CARDINALITY_THRESHOLD = 0.95
# Minimum number of rows before raising a high-cardinality warning.
_HIGH_CARDINALITY_MIN_ROWS = 50


def analyze_schema(
    df: pd.DataFrame,
    datetime_column_hints: Optional[List[str]] = None,
) -> SchemaReport:
    """Inspect column dtypes and surface potential type mismatches.

    Parameters
    ----------
    df:
        The DataFrame to analyse.
    datetime_column_hints:
        Column names the caller believes are datetime. If the column is still
        stored as object/string the checker will flag it regardless of the
        heuristic threshold.

    Returns
    -------
    SchemaReport
        Contains the dtype map and a list of :class:`SchemaIssue` objects.
    """
    hints = set(datetime_column_hints or [])
    dtype_map = {col: str(df[col].dtype) for col in df.columns}
    issues: List[SchemaIssue] = []

    obj_cols = categorical_columns(df)
    num_cols = set(numeric_columns(df))

    for col in obj_cols:
        series = df[col]
        n_rows = len(series.dropna())
        if n_rows == 0:
            continue

        # --- numeric stored as string ---
        if is_likely_numeric_string(series):
            issues.append(
                SchemaIssue(
                    column=col,
                    issue_type="numeric_as_string",
                    description=(
                        f"Column '{col}' has dtype object but its values look "
                        "numeric. Consider casting with pd.to_numeric()."
                    ),
                    suggestion=f"pd.to_numeric(df['{col}'], errors='coerce')",
                )
            )
            continue  # don't double-report

        # --- datetime stored as string ---
        is_hinted = col in hints
        if is_hinted or is_likely_datetime_string(series):
            issues.append(
                SchemaIssue(
                    column=col,
                    issue_type="datetime_as_string",
                    description=(
                        f"Column '{col}' has dtype object but its values look "
                        "like datetimes. Consider parsing with pd.to_datetime()."
                    ),
                    suggestion=f"pd.to_datetime(df['{col}'], errors='coerce')",
                )
            )
            continue

        # --- high-cardinality object column ---
        n_unique = series.nunique(dropna=True)
        if (
            n_rows >= _HIGH_CARDINALITY_MIN_ROWS
            and n_unique / n_rows >= _HIGH_CARDINALITY_THRESHOLD
        ):
            issues.append(
                SchemaIssue(
                    column=col,
                    issue_type="high_cardinality_object",
                    description=(
                        f"Column '{col}' is object dtype with {n_unique} unique "
                        f"values across {n_rows} non-null rows "
                        f"({n_unique / n_rows:.0%} uniqueness). "
                        "This may be a free-text or ID column."
                    ),
                    suggestion="Consider dropping or encoding this column if it is an identifier.",
                )
            )

    # --- constant columns (any dtype) ---
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            issues.append(
                SchemaIssue(
                    column=col,
                    issue_type="constant_column",
                    description=(
                        f"Column '{col}' contains only one distinct value "
                        "(including NaN). It carries no information."
                    ),
                    suggestion=f"Consider dropping: df.drop(columns=['{col}'])",
                )
            )

    return SchemaReport(column_dtypes=dtype_map, issues=issues)
