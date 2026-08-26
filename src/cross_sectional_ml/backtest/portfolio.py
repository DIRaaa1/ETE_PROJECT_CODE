"""Equal-notional AlphaMark-style portfolio selection and accounting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

from cross_sectional_ml.io import normalize_code_series, normalize_datetime_series

PortfolioMode = Literal["signed", "long_only"]
MODES: tuple[PortfolioMode, ...] = ("signed", "long_only")
QUANTILES: tuple[float, ...] = (1.0, 0.75, 0.50, 0.25)


def _canonical_quantile(quantile: float) -> float:
    for candidate in QUANTILES:
        if np.isclose(float(quantile), candidate, rtol=0.0, atol=1e-12):
            return candidate
    raise ValueError(f"quantile must be one of {QUANTILES}, got {quantile!r}")


def _validate_mode(mode: str) -> PortfolioMode:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return mode  # type: ignore[return-value]


def select_day(
    day: pd.DataFrame,
    *,
    mode: PortfolioMode,
    quantile: float,
    score_column: str = "score",
    code_column: str = "Code",
) -> pd.DataFrame:
    """Select one cumulative quantile with a deterministic Code tie-break.

    Signed portfolios rank finite non-zero scores by absolute magnitude.
    Long-only portfolios retain strictly positive scores and rank by score.
    The selected count is ``ceil(quantile * eligible_count)``.
    """

    mode = _validate_mode(mode)
    quantile = _canonical_quantile(quantile)
    missing = {score_column, code_column}.difference(day.columns)
    if missing:
        raise ValueError(f"missing selection columns: {sorted(missing)}")

    work = day.copy()
    work[code_column] = normalize_code_series(work[code_column])
    if work[code_column].isna().any() or work[code_column].eq("").any():
        raise ValueError(f"{code_column} contains missing or empty values")
    scores = pd.to_numeric(work[score_column], errors="raise").to_numpy(dtype=float)
    finite = np.isfinite(scores)
    if mode == "signed":
        eligible_mask = finite & (scores != 0.0)
        position = np.sign(scores)
        strength = np.abs(scores)
    else:
        eligible_mask = finite & (scores > 0.0)
        position = np.ones(len(work), dtype=float)
        strength = scores

    eligible = work.loc[eligible_mask].copy()
    eligible["_position"] = position[eligible_mask]
    eligible["_strength"] = strength[eligible_mask]
    # Stable second sort preserves ascending Code order within strength ties.
    eligible = eligible.sort_values(code_column, kind="mergesort")
    eligible = eligible.sort_values("_strength", ascending=False, kind="mergesort")
    count = int(np.ceil(quantile * len(eligible)))
    selected = eligible.iloc[:count].drop(columns="_strength")
    return selected.rename(columns={"_position": "position"}).reset_index(drop=True)


def _normalise_panel(
    panel: pd.DataFrame,
    *,
    date_column: str,
    code_column: str,
    score_column: str,
    outcome_column: str,
) -> pd.DataFrame:
    required = {date_column, code_column, score_column, outcome_column}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"missing backtest columns: {sorted(missing)}")
    out = panel.copy()
    out[date_column] = normalize_datetime_series(out[date_column])
    out[code_column] = normalize_code_series(out[code_column])
    out[score_column] = pd.to_numeric(out[score_column], errors="raise")
    out[outcome_column] = pd.to_numeric(out[outcome_column], errors="raise")
    if out[date_column].isna().any():
        raise ValueError(f"{date_column} contains missing values")
    if out[code_column].isna().any() or out[code_column].eq("").any():
        raise ValueError(f"{code_column} contains missing or empty values")
    if out.duplicated([date_column, code_column]).any():
        raise ValueError("backtest panel contains duplicate Date/Code keys")
    return out.sort_values([date_column, code_column], kind="mergesort").reset_index(drop=True)


def portfolio_day(
    day: pd.DataFrame,
    *,
    mode: PortfolioMode,
    quantile: float,
    score_column: str = "score",
    outcome_column: str = "return",
    code_column: str = "Code",
) -> dict[str, float | int]:
    """Compute one day's portfolio statistics under native missing accounting.

    An unavailable outcome contributes zero PnL, but the selected name keeps
    one unit of notional and is not replaced.  Missing and exactly zero returns
    are excluded from the hit-rate denominator.
    """

    if outcome_column not in day.columns:
        raise ValueError(f"missing outcome column: {outcome_column!r}")
    selected = select_day(
        day,
        mode=mode,
        quantile=quantile,
        score_column=score_column,
        code_column=code_column,
    )
    positions = selected["position"].to_numpy(dtype=float)
    realised = pd.to_numeric(selected[outcome_column], errors="raise").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    observed = np.isfinite(realised)
    contributions = np.where(observed, positions * realised, 0.0)

    selected_count = len(selected)
    notional = float(selected_count)
    pnl = float(contributions.sum())
    hit_eligible = observed & (realised != 0.0)
    hit_count = int(np.count_nonzero(hit_eligible & (contributions > 0.0)))
    hit_denominator = int(np.count_nonzero(hit_eligible))
    long_count = int(np.count_nonzero(positions > 0.0))

    return {
        "pnl": pnl,
        "notional": notional,
        "ppd": pnl / notional if notional > 0.0 else 0.0,
        "hit_count": hit_count,
        "hit_denominator": hit_denominator,
        "hit_ratio": hit_count / hit_denominator if hit_denominator else 0.0,
        "long_count": long_count,
        "position_count": selected_count,
        "long_share": long_count / selected_count if selected_count else 0.0,
        "selected_count": selected_count,
        "observed_return_count": int(np.count_nonzero(observed)),
        "missing_return_count": int(np.count_nonzero(~observed)),
    }


def backtest_daily(
    panel: pd.DataFrame,
    *,
    mode: PortfolioMode,
    quantile: float,
    score_column: str = "score",
    outcome_column: str = "return",
    date_column: str = "Date",
    code_column: str = "Code",
) -> pd.DataFrame:
    """Evaluate one model/mode/quantile and return one row per trading day."""

    mode = _validate_mode(mode)
    quantile = _canonical_quantile(quantile)
    work = _normalise_panel(
        panel,
        date_column=date_column,
        code_column=code_column,
        score_column=score_column,
        outcome_column=outcome_column,
    )
    rows: list[dict[str, object]] = []
    for date, day in work.groupby(date_column, sort=True, observed=True):
        row: dict[str, object] = {
            date_column: date,
            "mode": mode,
            "quantile": quantile,
        }
        row.update(
            portfolio_day(
                day,
                mode=mode,
                quantile=quantile,
                score_column=score_column,
                outcome_column=outcome_column,
                code_column=code_column,
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def backtest_grid(
    panel: pd.DataFrame,
    *,
    score_column: str = "score",
    outcome_column: str = "return",
    date_column: str = "Date",
    code_column: str = "Code",
    modes: Iterable[PortfolioMode] = MODES,
    quantiles: Iterable[float] = QUANTILES,
) -> pd.DataFrame:
    """Evaluate the complete signed/long-only by qr100/75/50/25 grid."""

    mode_values = tuple(modes)
    quantile_values = tuple(quantiles)
    pieces = [
        backtest_daily(
            panel,
            mode=mode,
            quantile=quantile,
            score_column=score_column,
            outcome_column=outcome_column,
            date_column=date_column,
            code_column=code_column,
        )
        for mode in mode_values
        for quantile in quantile_values
    ]
    if not pieces:
        return pd.DataFrame()
    # The loop order is the public order: signed, long-only, then qr100 to qr25.
    return pd.concat(pieces, ignore_index=True)


def summarise_daily(
    daily: pd.DataFrame,
    *,
    date_column: str = "Date",
) -> pd.Series:
    """Recompute AlphaMark summary metrics from daily sufficient statistics."""

    required = {
        date_column,
        "pnl",
        "notional",
        "hit_count",
        "hit_denominator",
        "long_count",
        "position_count",
        "selected_count",
    }
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"missing daily summary columns: {sorted(missing)}")
    if daily.empty:
        raise ValueError("cannot summarise an empty daily table")

    work = daily.copy()
    work[date_column] = normalize_datetime_series(work[date_column])
    if work[date_column].duplicated().any():
        raise ValueError("summarise_daily expects one portfolio row per date")
    numeric_columns = required.difference({date_column})
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
        if not np.isfinite(work[column].to_numpy(dtype=float)).all():
            raise ValueError(f"daily column {column!r} must be finite")

    pnl_series = work["pnl"].to_numpy(dtype=float)
    pnl = float(pnl_series.sum())
    notional = float(work["notional"].sum())
    if len(pnl_series) > 1:
        standard_deviation = float(np.std(pnl_series, ddof=1))
        sharpe = (
            float(np.sqrt(252.0) * np.mean(pnl_series) / standard_deviation)
            if standard_deviation > 0.0
            else 0.0
        )
    else:
        sharpe = 0.0
    if notional == 0.0:
        sharpe = 0.0

    hit_count = int(work["hit_count"].sum())
    hit_denominator = int(work["hit_denominator"].sum())
    long_count = int(work["long_count"].sum())
    position_count = int(work["position_count"].sum())
    selected_stock_days = int(work["selected_count"].sum())

    return pd.Series(
        {
            "pnl": pnl,
            "notional": notional,
            "ppd": pnl / notional if notional > 0.0 else 0.0,
            "sharpe_ratio": sharpe,
            "hit_ratio": hit_count / hit_denominator if hit_denominator else 0.0,
            "long_share": long_count / position_count if position_count else 0.0,
            "nr_instr": float(work["selected_count"].mean()),
            "nr_trades": selected_stock_days,
            "n_days": len(work),
            "cash_days": int(work["notional"].eq(0.0).sum()),
        },
        name="summary",
    )
