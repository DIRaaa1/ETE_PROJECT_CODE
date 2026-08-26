from __future__ import annotations

import pandas as pd

from .portfolio import summarise_daily


def portfolio_summary(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, mode, quantile), frame in daily.groupby(
        ["model", "mode", "quantile"], sort=False, observed=True
    ):
        row = {"period": period, "model": model, "mode": mode, "quantile": quantile}
        row.update(summarise_daily(frame).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def predictive_summary(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    summary = daily.groupby("model", sort=False, observed=True)[["ic", "rank_ic"]].mean().reset_index()
    summary.insert(0, "period", period)
    return summary
