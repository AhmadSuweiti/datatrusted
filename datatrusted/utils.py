"""
Internal utility functions shared across datatrusted modules.

Nothing in here is part of the public API.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Type-detection helpers
# ---------------------------------------------------------------------------

_NUMERIC_PATTERN = re.compile(
    r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$"
)

_DATETIME_PATTERNS = [
    # ISO-ish
    re.compile(r"^\d{4}-\d{2}-\d{2}"),
    # US formats
    re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}"),
    re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}"),
]


def is_likely_numeric_string(series: pd.Series, sample_size: int = 200) -> bool:
    """Return True if a non-numeric Series contains mostly numeric-looking strings.

    Samples up to `sample_size` non-null values and checks whether ≥ 90 % of
    them match a numeric pattern.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.sample(min(sample_size, len(non_null)), random_state=42).astype(str)
    matches = sample.apply(lambda v: bool(_NUMERIC_PATTERN.match(v))).sum()
    return (matches / len(sample)) >= 0.90


def is_likely_datetime_string(series: pd.Series, sample_size: int = 200) -> bool:
    """Return True if a non-datetime Series contains mostly datetime-looking strings."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False

    sample = non_null.sample(min(sample_size, len(non_null)), random_state=42).astype(str)

    def _matches_any(v: str) -> bool:
        return any(p.match(v) for p in _DATETIME_PATTERNS)

    matches = sample.apply(_matches_any).sum()
    return (matches / len(sample)) >= 0.80


def safe_sample(series: pd.Series, n: int = 5) -> List[Any]:
    """Return up to `n` non-null sample values from a Series as a plain list."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    return non_null.sample(min(n, len(non_null)), random_state=0).tolist()


# ---------------------------------------------------------------------------
# Column-classification helpers
# ---------------------------------------------------------------------------


def numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return column names that pandas considers numeric."""
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> List[str]:
    """Return object / category / boolean columns."""
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> List[str]:
    """Return columns with a datetime64 dtype."""
    return df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def pct_str(value: float, decimals: int = 1) -> str:
    """Format a fraction (0–1) as a percentage string, e.g. '12.3%'."""
    return f"{value * 100:.{decimals}f}%"


def truncate_list(items: list, max_items: int = 10) -> list:
    """Return at most `max_items` elements with no truncation indicator."""
    return items[:max_items]


def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    """Return 'N word' with correct singular/plural form."""
    word = singular if count == 1 else (plural or singular + "s")
    return f"{count} {word}"
