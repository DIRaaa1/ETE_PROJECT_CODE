"""Daily prediction dependence and the prespecified eight-model ensemble."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from cross_sectional_ml.io import normalize_code_series, normalize_datetime_series

from .predictions import EIGHT_MODELS


def _prepare_wide_panel(
    panel: pd.DataFrame,
    *,
    model_columns: Sequence[str],
    date_column: str,
    code_column: str,
) -> pd.DataFrame:
    if tuple(model_columns) != EIGHT_MODELS:
        raise ValueError(f"model_columns must be the fixed dissertation order {EIGHT_MODELS}")
    required = {date_column, code_column, *model_columns}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"missing ensemble columns: {sorted(missing)}")

    out = panel[[date_column, code_column, *model_columns]].copy()
    out[date_column] = normalize_datetime_series(out[date_column])
    out[code_column] = normalize_code_series(out[code_column])
    if out[date_column].isna().any():
        raise ValueError(f"{date_column} contains missing values")
    if out[code_column].isna().any() or out[code_column].eq("").any():
        raise ValueError(f"{code_column} contains missing or empty values")
    if out.duplicated([date_column, code_column]).any():
        raise ValueError("ensemble panel contains duplicate Date/Code keys")
    for model in model_columns:
        values = pd.to_numeric(out[model], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{model} predictions must all be finite")
        out[model] = values
    return out.sort_values([date_column, code_column], kind="mergesort").reset_index(drop=True)


def _daily_ranks_and_scale(
    day: pd.DataFrame,
    *,
    model_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.Series]:
    if len(day) < 2:
        raise ValueError("each date needs at least two stocks for rank dependence")
    ranks = day[list(model_columns)].rank(axis=0, method="average")
    scale = ranks.std(axis=0, ddof=1)
    if not np.isfinite(scale.to_numpy(dtype=float)).all() or scale.le(0.0).any():
        constant = scale.index[scale.le(0.0) | scale.isna()].tolist()
        raise ValueError(f"constant cross-sectional model ranks: {constant}")
    return ranks, scale


def average_daily_spearman(
    panel: pd.DataFrame,
    *,
    model_columns: Sequence[str] = EIGHT_MODELS,
    date_column: str = "Date",
    code_column: str = "Code",
) -> pd.DataFrame:
    """Return the equal-date mean of daily cross-sectional Spearman matrices."""

    work = _prepare_wide_panel(
        panel,
        model_columns=model_columns,
        date_column=date_column,
        code_column=code_column,
    )
    if work.empty:
        raise ValueError("cannot compute prediction dependence for an empty panel")
    total = np.zeros((len(model_columns), len(model_columns)), dtype=float)
    date_count = 0
    for _, day in work.groupby(date_column, sort=True, observed=True):
        ranks, _ = _daily_ranks_and_scale(day, model_columns=model_columns)
        correlation = ranks.corr(method="pearson").to_numpy(dtype=float)
        if not np.isfinite(correlation).all():
            raise ValueError("daily Spearman matrix is not finite")
        total += correlation
        date_count += 1
    return pd.DataFrame(
        total / date_count,
        index=list(model_columns),
        columns=list(model_columns),
    )


def rank_zscore_ensemble(
    panel: pd.DataFrame,
    *,
    model_columns: Sequence[str] = EIGHT_MODELS,
    output_column: str = "ensemble_score",
    date_column: str = "Date",
    code_column: str = "Code",
) -> pd.DataFrame:
    """Build the equal-weight ensemble from daily average-tie rank z-scores."""

    if output_column in {date_column, code_column, *model_columns}:
        raise ValueError("output_column collides with an input column")
    work = _prepare_wide_panel(
        panel,
        model_columns=model_columns,
        date_column=date_column,
        code_column=code_column,
    )
    pieces: list[pd.DataFrame] = []
    for _, day in work.groupby(date_column, sort=True, observed=True):
        ranks, scale = _daily_ranks_and_scale(day, model_columns=model_columns)
        standardised = (ranks - ranks.mean(axis=0)) / scale
        scores = standardised.mean(axis=1).to_numpy(dtype=float)
        piece = day[[date_column, code_column]].copy()
        piece[output_column] = scores
        pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=[date_column, code_column, output_column])
    return pd.concat(pieces, ignore_index=True)
