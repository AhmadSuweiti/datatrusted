"""Shared fixtures for datatrusted tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """A small, clean DataFrame with numeric and categorical columns."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "age": [25, 32, 45, 28, 39],
            "salary": [50_000, 75_000, 90_000, 60_000, 80_000],
            "city": ["London", "Paris", "Berlin", "Madrid", "Rome"],
            "active": [True, True, False, True, False],
        }
    )


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """DataFrame with deliberate missing values."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "age": [25, None, 45, None, 39, None, None, None, None, None],  # 60 % missing
            "salary": [50_000, 75_000, None, 60_000, 80_000, 70_000, None, 55_000, 65_000, 72_000],
            "city": ["London", "Paris", None, "Madrid", None, "Rome", "Athens", None, "Lisbon", "Oslo"],
        }
    )


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """DataFrame with full-row duplicates and a non-unique ID."""
    return pd.DataFrame(
        {
            "id": [1, 2, 2, 3, 4, 4, 5],
            "value": [10, 20, 20, 30, 40, 40, 50],
        }
    )


@pytest.fixture
def classification_df() -> pd.DataFrame:
    """DataFrame suitable for testing a classification target."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(5, 2, n),
            "label": rng.choice([0, 1], n, p=[0.8, 0.2]),  # imbalanced
        }
    )
