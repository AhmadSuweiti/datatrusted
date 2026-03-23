"""
Duplicate row and key detection.

Checks for exact full-row duplicates and, optionally, duplicate values in
designated ID or key columns.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from datatrust.models import DuplicateInfo


def analyze_duplicates(
    df: pd.DataFrame,
    id_columns: Optional[List[str]] = None,
) -> DuplicateInfo:
    """Count duplicate rows and flag duplicate values in ID columns.

    Parameters
    ----------
    df:
        DataFrame to analyse.
    id_columns:
        Column names that should be unique identifiers (e.g. ``['user_id']``).
        For each, the number of rows whose value appears more than once is
        reported.

    Returns
    -------
    DuplicateInfo
    """
    total_rows = len(df)

    if total_rows == 0:
        return DuplicateInfo(
            full_row_duplicates=0,
            total_rows=0,
            duplicate_pct=0.0,
        )

    full_dupes = int(df.duplicated().sum())
    duplicate_pct = full_dupes / total_rows

    id_col_dupes: dict[str, int] = {}
    for col in id_columns or []:
        if col not in df.columns:
            continue
        # Count rows whose key value appears more than once
        counts = df[col].value_counts()
        duped_values = counts[counts > 1]
        # Total rows involved in a duplicate key situation
        id_col_dupes[col] = int(duped_values.sum()) - len(duped_values)

    return DuplicateInfo(
        full_row_duplicates=full_dupes,
        total_rows=total_rows,
        duplicate_pct=duplicate_pct,
        id_column_duplicates=id_col_dupes,
    )
