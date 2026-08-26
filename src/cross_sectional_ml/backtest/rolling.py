"""Chronological F1-2024/F2-2025 stitching with metric recomputation."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from cross_sectional_ml.io import normalize_datetime_series

from .portfolio import summarise_daily


def _prepare_year(
    daily: pd.DataFrame,
    *,
    expected_year: int,
    date_column: str,
) -> pd.DataFrame:
    if daily.empty:
        raise ValueError(f"{expected_year} daily table is empty")
    if date_column not in daily.columns:
        raise ValueError(f"missing date column: {date_column!r}")
    out = daily.copy()
    out[date_column] = normalize_datetime_series(out[date_column])
    years = set(out[date_column].dt.year.unique().tolist())
    if years != {expected_year}:
        raise ValueError(f"expected only {expected_year} rows, found years={sorted(years)}")
    return out


def combine_rolling_years(
    daily_2024: pd.DataFrame,
    daily_2025: pd.DataFrame,
    *,
    date_column: str = "Date",
    group_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame]:
    """Concatenate the two test folds and recompute, never average, summaries.

    With no ``group_columns`` the inputs represent one portfolio.  For a full
    grid, pass identifiers such as ``("model", "mode", "quantile")`` and a
    summary DataFrame is returned with one independently recomputed row per
    group.
    """

    first = _prepare_year(daily_2024, expected_year=2024, date_column=date_column)
    second = _prepare_year(daily_2025, expected_year=2025, date_column=date_column)
    if set(first.columns) != set(second.columns):
        raise ValueError("2024 and 2025 daily tables must have the same columns")
    second = second[first.columns]

    groups = list(group_columns)
    missing_groups = set(groups).difference(first.columns)
    if missing_groups:
        raise ValueError(f"missing group columns: {sorted(missing_groups)}")
    if groups:
        first_groups = set(first[groups].itertuples(index=False, name=None))
        second_groups = set(second[groups].itertuples(index=False, name=None))
        if first_groups != second_groups:
            raise ValueError("2024 and 2025 must contain the same portfolio groups")

    combined = pd.concat([first, second], ignore_index=True)
    key_columns = [*groups, date_column]
    if combined.duplicated(key_columns).any():
        raise ValueError(f"duplicate rolling rows for keys {key_columns}")
    combined = combined.sort_values(key_columns, kind="mergesort").reset_index(drop=True)

    if not groups:
        return combined, summarise_daily(combined, date_column=date_column)

    summary_rows: list[dict[str, object]] = []
    group_key: str | list[str] = groups[0] if len(groups) == 1 else groups
    for values, frame in combined.groupby(group_key, sort=True, observed=True):
        value_tuple = values if isinstance(values, tuple) else (values,)
        row = dict(zip(groups, value_tuple, strict=True))
        row.update(summarise_daily(frame, date_column=date_column).to_dict())
        summary_rows.append(row)
    return combined, pd.DataFrame(summary_rows)
