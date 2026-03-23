"""
Join integrity checks.

Before merging two DataFrames, verify that the join keys behave as expected:
no unexpected duplicates, no unmatched rows, and no accidental many-to-many
fan-out.
"""

from __future__ import annotations

from typing import List, Union

import pandas as pd

from datatrust.models import JoinReport


def check_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, List[str]],
) -> JoinReport:
    """Analyse the integrity of a potential join between *left* and *right*.

    Parameters
    ----------
    left:
        Left DataFrame.
    right:
        Right DataFrame.
    on:
        Column name(s) to join on. Can be a single string or a list.

    Returns
    -------
    JoinReport
    """
    if isinstance(on, str):
        on = [on]

    warnings: List[str] = []

    # ------------------------------------------------------------------
    # Validate that all key columns exist in both frames
    # ------------------------------------------------------------------
    missing_left = [c for c in on if c not in left.columns]
    missing_right = [c for c in on if c not in right.columns]

    if missing_left:
        warnings.append(
            f"Key column(s) {missing_left} not found in left DataFrame."
        )
    if missing_right:
        warnings.append(
            f"Key column(s) {missing_right} not found in right DataFrame."
        )

    valid_keys = [c for c in on if c not in missing_left and c not in missing_right]

    if not valid_keys:
        return JoinReport(
            on=on,
            left_shape=left.shape,
            right_shape=right.shape,
            left_duplicates=0,
            right_duplicates=0,
            unmatched_left=0,
            unmatched_right=0,
            is_many_to_many=False,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Duplicate key detection
    # ------------------------------------------------------------------
    left_key_counts = left[valid_keys].value_counts()
    right_key_counts = right[valid_keys].value_counts()

    left_dup_keys = left_key_counts[left_key_counts > 1]
    right_dup_keys = right_key_counts[right_key_counts > 1]

    # Rows involved in duplicate keys
    left_duplicates = int(left_dup_keys.sum()) - len(left_dup_keys)
    right_duplicates = int(right_dup_keys.sum()) - len(right_dup_keys)

    if left_duplicates > 0:
        warnings.append(
            f"Left DataFrame has {left_duplicates} row(s) with duplicate "
            f"join key values on {valid_keys}. "
            "A left join may produce unexpected fan-out."
        )
    if right_duplicates > 0:
        warnings.append(
            f"Right DataFrame has {right_duplicates} row(s) with duplicate "
            f"join key values on {valid_keys}. "
            "A right join may produce unexpected fan-out."
        )

    # Many-to-many: both sides have duplicate keys
    is_many_to_many = left_duplicates > 0 and right_duplicates > 0
    if is_many_to_many:
        warnings.append(
            "Both DataFrames have duplicate join keys — this is a many-to-many "
            "join and will multiply rows. The result will have more rows than "
            "either input."
        )

    # ------------------------------------------------------------------
    # Unmatched key detection
    # ------------------------------------------------------------------
    left_keys = _key_set(left, valid_keys)
    right_keys = _key_set(right, valid_keys)

    only_in_left = left_keys - right_keys
    only_in_right = right_keys - left_keys

    unmatched_left = int(
        left[valid_keys].apply(tuple, axis=1).isin(only_in_left).sum()
    )
    unmatched_right = int(
        right[valid_keys].apply(tuple, axis=1).isin(only_in_right).sum()
    )

    if unmatched_left > 0:
        warnings.append(
            f"{unmatched_left} row(s) in the left DataFrame have no matching "
            f"key in the right DataFrame ({len(only_in_left)} distinct key(s) unmatched). "
            "An inner join will drop these rows."
        )
    if unmatched_right > 0:
        warnings.append(
            f"{unmatched_right} row(s) in the right DataFrame have no matching "
            f"key in the left DataFrame ({len(only_in_right)} distinct key(s) unmatched). "
            "An inner join will drop these rows."
        )

    return JoinReport(
        on=on,
        left_shape=left.shape,
        right_shape=right.shape,
        left_duplicates=left_duplicates,
        right_duplicates=right_duplicates,
        unmatched_left=unmatched_left,
        unmatched_right=unmatched_right,
        is_many_to_many=is_many_to_many,
        warnings=warnings,
    )


def _key_set(df: pd.DataFrame, keys: List[str]) -> set:
    """Return the set of distinct key tuples present in *df*."""
    return set(map(tuple, df[keys].values.tolist()))
