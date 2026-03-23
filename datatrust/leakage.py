"""
Heuristic data leakage detection.

These checks produce *hints*, not definitive findings. All results should be
reviewed by a human before acting on them. The goal is to surface columns
worth investigating, not to auto-remove anything.

Checks performed:
1. **Suspicious name** — columns whose names suggest they encode the outcome
   (e.g. ``status``, ``label``, ``outcome``, ``approved``).
2. **Near-perfect target correlation** — numeric columns whose absolute
   Pearson correlation with the target exceeds a high threshold.
3. **High uniqueness relative to target** — object columns where every row
   maps near-uniquely to a target value (likely a post-event identifier).
"""

from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

from datatrust.models import LeakageHint
from datatrust.utils import numeric_columns

# Column name fragments that suggest the column encodes the outcome.
_SUSPICIOUS_NAME_PATTERNS = [
    r"\btarget\b",
    r"\blabel\b",
    r"\boutcome\b",
    r"\bstatus\b",
    r"\bresult\b",
    r"\bclosed\b",
    r"\bapproved\b",
    r"\brejected\b",
    r"\bchurned?\b",
    r"\bconverted?\b",
    r"\bdefaulted?\b",
    r"\bfraud\b",
    r"\bflag\b",
    r"\bresponse\b",
    r"\bclass\b",
    r"\bscore\b",
]

_SUSPICIOUS_RE = re.compile(
    "|".join(_SUSPICIOUS_NAME_PATTERNS),
    flags=re.IGNORECASE,
)

# Pearson |r| above this is flagged as potential leakage.
_CORRELATION_THRESHOLD = 0.95

# For object columns: if the mean number of target values per unique key is
# close to 1, the column is effectively a near-unique ID that may leak.
_NEAR_UNIQUE_RATIO_THRESHOLD = 0.98


def analyze_leakage(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> List[LeakageHint]:
    """Run heuristic leakage checks and return a list of hints.

    Parameters
    ----------
    df:
        DataFrame to inspect.
    target_column:
        Optional target column name. Required for correlation-based checks.

    Returns
    -------
    list of LeakageHint
    """
    hints: List[LeakageHint] = []

    columns_to_check = [c for c in df.columns if c != target_column]

    # ------------------------------------------------------------------
    # 1. Suspicious column names
    # ------------------------------------------------------------------
    for col in columns_to_check:
        if _SUSPICIOUS_RE.search(col):
            hints.append(
                LeakageHint(
                    column=col,
                    hint_type="suspicious_name",
                    description=(
                        f"Column '{col}' has a name that may indicate it encodes "
                        "the outcome (e.g. 'status', 'label', 'result'). "
                        "Verify that this column is not derived from the target "
                        "or represents post-event information."
                    ),
                    severity="medium",
                )
            )

    if target_column is None or target_column not in df.columns:
        return hints

    target_series = df[target_column]

    # ------------------------------------------------------------------
    # 2. Near-perfect numeric correlation with target
    # ------------------------------------------------------------------
    if pd.api.types.is_numeric_dtype(target_series):
        num_cols = numeric_columns(df)
        for col in num_cols:
            if col == target_column:
                continue
            try:
                r = float(df[col].corr(target_series))
            except Exception:
                continue
            if pd.isna(r):
                continue
            if abs(r) >= _CORRELATION_THRESHOLD:
                hints.append(
                    LeakageHint(
                        column=col,
                        hint_type="high_target_correlation",
                        description=(
                            f"Column '{col}' has a Pearson correlation of "
                            f"{r:.3f} with the target '{target_column}'. "
                            "This may indicate the column is derived from or "
                            "directly encodes the target."
                        ),
                        severity="high",
                    )
                )

    # ------------------------------------------------------------------
    # 3. Object columns that near-uniquely identify target values
    # ------------------------------------------------------------------
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in obj_cols:
        if col == target_column:
            continue
        try:
            # Fraction of (col, target) pairs that are unique
            combined = df[[col, target_column]].dropna()
            if len(combined) == 0:
                continue
            n_combined_unique = combined.drop_duplicates().shape[0]
            n_col_unique = combined[col].nunique()
            if n_col_unique == 0:
                continue
            # If each unique value of col maps to almost exactly one target value,
            # the column could be a post-event code leaking the outcome.
            ratio = n_combined_unique / len(combined)
            if ratio >= _NEAR_UNIQUE_RATIO_THRESHOLD:
                hints.append(
                    LeakageHint(
                        column=col,
                        hint_type="near_unique_key",
                        description=(
                            f"Column '{col}' has {ratio:.1%} unique (col, target) "
                            f"combinations — nearly every row is distinct. "
                            "This may be a post-event identifier or derived field."
                        ),
                        severity="medium",
                    )
                )
        except Exception:
            continue

    return hints
