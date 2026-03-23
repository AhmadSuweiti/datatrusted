"""
Target column analysis.

When a target/label column is specified, this module checks for missing
labels, analyses class distributions for classification targets, and flags
potential issues like severe class imbalance or unexpectedly high cardinality.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from datatrust.models import TargetInfo

# A target with ≤ this many unique values is treated as a classification target.
_CLASSIFICATION_MAX_UNIQUE = 20
# Warn when the imbalance ratio (majority / minority class) exceeds this.
_IMBALANCE_RATIO_WARN = 5.0
# Warn when unique count is suspiciously high relative to row count.
_CARDINALITY_WARN_THRESHOLD = 0.50


def analyze_target(df: pd.DataFrame, target_column: str) -> Optional[TargetInfo]:
    """Analyse the target/label column.

    Parameters
    ----------
    df:
        DataFrame containing the target column.
    target_column:
        Name of the column to analyse.

    Returns
    -------
    TargetInfo or None
        None is returned when *target_column* is not present in *df*.
    """
    if target_column not in df.columns:
        return None

    series = df[target_column]
    total = len(series)
    missing_count = int(series.isnull().sum())
    missing_pct = missing_count / total if total > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))

    is_likely_classification = unique_count <= _CLASSIFICATION_MAX_UNIQUE

    # Cardinality warning: target has more than 50 % unique values — it might
    # be a continuous variable or an ID column used by mistake.
    cardinality_warning = (
        unique_count / total > _CARDINALITY_WARN_THRESHOLD if total > 0 else False
    )

    class_counts: dict[str, int] = {}
    imbalance_ratio: Optional[float] = None

    if is_likely_classification:
        vc = series.dropna().value_counts()
        class_counts = {str(k): int(v) for k, v in vc.items()}

        if len(vc) >= 2:
            majority = int(vc.iloc[0])
            minority = int(vc.iloc[-1])
            if minority > 0:
                imbalance_ratio = majority / minority

    return TargetInfo(
        column=target_column,
        missing_count=missing_count,
        missing_pct=missing_pct,
        unique_count=unique_count,
        is_likely_classification=is_likely_classification,
        class_counts=class_counts,
        imbalance_ratio=imbalance_ratio,
        cardinality_warning=cardinality_warning,
    )
