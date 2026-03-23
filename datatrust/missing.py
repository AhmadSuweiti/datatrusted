"""
Missing value analysis.

Computes per-column and overall missing-value statistics and flags columns
that exceed a configurable missingness threshold.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from datatrust.models import MissingInfo

_DEFAULT_THRESHOLD = 0.05  # 5 %


def analyze_missing(
    df: pd.DataFrame,
    threshold: float = _DEFAULT_THRESHOLD,
) -> MissingInfo:
    """Compute missing-value statistics for every column.

    Parameters
    ----------
    df:
        DataFrame to analyse.
    threshold:
        Fraction (0–1). Columns whose missing percentage exceeds this value
        are listed in :attr:`MissingInfo.columns_above_threshold`.

    Returns
    -------
    MissingInfo
    """
    if df.empty:
        return MissingInfo(
            total_missing=0,
            total_cells=0,
            threshold=threshold,
        )

    null_counts = df.isnull().sum()
    total_rows = len(df)
    total_cells = total_rows * len(df.columns)
    total_missing = int(null_counts.sum())

    missing_by_column: dict[str, int] = {}
    missing_pct_by_column: dict[str, float] = {}
    columns_above: List[str] = []

    for col in df.columns:
        count = int(null_counts[col])
        pct = count / total_rows if total_rows > 0 else 0.0
        missing_by_column[col] = count
        missing_pct_by_column[col] = pct
        if pct > threshold:
            columns_above.append(col)

    return MissingInfo(
        total_missing=total_missing,
        total_cells=total_cells,
        missing_by_column=missing_by_column,
        missing_pct_by_column=missing_pct_by_column,
        columns_above_threshold=columns_above,
        threshold=threshold,
    )


def missing_summary_table(info: MissingInfo) -> pd.DataFrame:
    """Return a DataFrame summarising per-column missing statistics.

    Useful for quick inspection in a notebook.
    """
    rows = [
        {
            "column": col,
            "missing_count": info.missing_by_column[col],
            "missing_pct": info.missing_pct_by_column[col],
            "above_threshold": col in info.columns_above_threshold,
        }
        for col in info.missing_by_column
    ]
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)
