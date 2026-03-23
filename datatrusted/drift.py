"""
Train/test distribution drift detection.

Compares two DataFrames (typically a training split and a test/validation
split) and reports columns where the distributions have shifted
meaningfully.

Numeric drift  — mean shift measured in units of training standard deviations
                 (similar to a simple effect size). Flagged when |shift| > 0.5.
Categorical drift — Total Variation Distance (TVD) between the training and
                    test frequency distributions. Flagged when TVD > 0.10.

Neither metric requires scipy or statsmodels. The goal is a practical,
interpretable signal rather than a formal statistical test.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from datatrusted.models import DriftInfo, DriftReport
from datatrusted.utils import categorical_columns, numeric_columns

# Numeric: flag when mean shift exceeds this many training std deviations.
_NUMERIC_SHIFT_THRESHOLD = 0.5

# Categorical: flag when Total Variation Distance exceeds this value.
_TVD_THRESHOLD = 0.10


def compare_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: Optional[str] = None,
) -> DriftReport:
    """Compare train and test DataFrames for distribution drift.

    Parameters
    ----------
    train_df:
        Training split.
    test_df:
        Test/validation split.
    target:
        Optional target column name. It is excluded from drift checks.

    Returns
    -------
    DriftReport
    """
    exclude = {target} if target else set()

    numeric_drifts = _check_numeric_drift(train_df, test_df, exclude)
    categorical_drifts = _check_categorical_drift(train_df, test_df, exclude)

    return DriftReport(
        train_shape=train_df.shape,
        test_shape=test_df.shape,
        numeric_drifts=numeric_drifts,
        categorical_drifts=categorical_drifts,
    )


# ---------------------------------------------------------------------------
# Numeric drift
# ---------------------------------------------------------------------------


def _check_numeric_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exclude: set,
) -> List[DriftInfo]:
    results: List[DriftInfo] = []

    # Only compare columns present in both frames
    train_num = set(numeric_columns(train)) - exclude
    test_num = set(numeric_columns(test)) - exclude
    common = train_num & test_num

    for col in sorted(common):
        train_vals = train[col].dropna()
        test_vals = test[col].dropna()

        if len(train_vals) < 2 or len(test_vals) < 2:
            continue

        train_mean = float(train_vals.mean())
        train_std = float(train_vals.std())
        test_mean = float(test_vals.mean())

        if train_std == 0:
            # Constant column in train — any test variation is drift
            shift = abs(test_mean - train_mean)
            drift_score = 1.0 if shift > 0 else 0.0
        else:
            raw_shift = (test_mean - train_mean) / train_std
            drift_score = min(abs(raw_shift) / 2.0, 1.0)  # normalise to [0,1]

        drift_detected = drift_score >= (_NUMERIC_SHIFT_THRESHOLD / 2.0)

        description = (
            f"Train mean={train_mean:.4g}, std={train_std:.4g}; "
            f"Test mean={test_mean:.4g}. "
        )
        if drift_detected:
            direction = "increased" if test_mean > train_mean else "decreased"
            description += f"Mean has {direction} by {abs(test_mean - train_mean):.4g}."
        else:
            description += "No significant drift detected."

        results.append(
            DriftInfo(
                column=col,
                column_type="numeric",
                drift_detected=drift_detected,
                drift_score=round(drift_score, 4),
                description=description,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Categorical drift
# ---------------------------------------------------------------------------


def _check_categorical_drift(
    train: pd.DataFrame,
    test: pd.DataFrame,
    exclude: set,
) -> List[DriftInfo]:
    results: List[DriftInfo] = []

    train_cat = set(categorical_columns(train)) - exclude
    test_cat = set(categorical_columns(test)) - exclude
    common = train_cat & test_cat

    for col in sorted(common):
        train_vals = train[col].dropna()
        test_vals = test[col].dropna()

        if len(train_vals) == 0 or len(test_vals) == 0:
            continue

        train_freq = train_vals.value_counts(normalize=True)
        test_freq = test_vals.value_counts(normalize=True)

        all_categories = set(train_freq.index) | set(test_freq.index)

        # Align both distributions over all categories (fill 0 for missing)
        train_aligned = train_freq.reindex(all_categories, fill_value=0.0)
        test_aligned = test_freq.reindex(all_categories, fill_value=0.0)

        # Total Variation Distance = 0.5 * sum(|p - q|)
        tvd = float(0.5 * (train_aligned - test_aligned).abs().sum())

        drift_detected = tvd >= _TVD_THRESHOLD

        train_cats = set(train_freq.index)
        test_cats = set(test_freq.index)
        missing_in_test = sorted(str(c) for c in train_cats - test_cats)
        unseen_in_test = sorted(str(c) for c in test_cats - train_cats)

        description = f"TVD={tvd:.4f}. "
        if missing_in_test:
            description += f"{len(missing_in_test)} train category(ies) absent in test. "
        if unseen_in_test:
            description += f"{len(unseen_in_test)} new category(ies) in test. "
        if not drift_detected:
            description += "No significant categorical drift detected."

        results.append(
            DriftInfo(
                column=col,
                column_type="categorical",
                drift_detected=drift_detected,
                drift_score=round(tvd, 4),
                description=description.strip(),
                missing_in_test=missing_in_test,
                unseen_in_test=unseen_in_test,
            )
        )

    return results
