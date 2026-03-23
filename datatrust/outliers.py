"""
IQR-based outlier detection.

Reports the number of potential outliers per numeric column using the
standard Tukey fence (Q1 - 1.5·IQR, Q3 + 1.5·IQR). Values outside
those bounds are flagged but never removed or modified.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from datatrust.models import OutlierInfo
from datatrust.utils import numeric_columns

_IQR_MULTIPLIER = 1.5


def analyze_outliers(df: pd.DataFrame) -> List[OutlierInfo]:
    """Run IQR outlier detection on every numeric column.

    Parameters
    ----------
    df:
        DataFrame to analyse.

    Returns
    -------
    list of OutlierInfo
        One entry per numeric column that has at least one outlier.
        Columns with no outliers are omitted from the result.
    """
    results: List[OutlierInfo] = []
    num_cols = numeric_columns(df)

    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 4:
            # Not enough data to compute meaningful quartiles
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        if iqr == 0:
            # Constant (non-null) column — no meaningful spread
            continue

        lower = q1 - _IQR_MULTIPLIER * iqr
        upper = q3 + _IQR_MULTIPLIER * iqr

        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())

        if count == 0:
            continue

        results.append(
            OutlierInfo(
                column=col,
                outlier_count=count,
                outlier_pct=count / len(series),
                lower_bound=lower,
                upper_bound=upper,
            )
        )

    return results
