from __future__ import annotations

import numpy as np
import pandas as pd


def correlation(x: pd.Series, y: pd.Series, *, rank: bool) -> float:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return 0.0
    if rank:
        pair = pair.rank(method="average")
    return float(pair["x"].corr(pair["y"]))


def daily_predictive_metrics(
    panel: pd.DataFrame,
    *,
    model: str,
    outcome_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, day in panel.groupby("Date", sort=True, observed=True):
        rows.append(
            {
                "Date": date,
                "model": model,
                "ic": correlation(day["score"], day[outcome_column], rank=False),
                "rank_ic": correlation(day["score"], day[outcome_column], rank=True),
            }
        )
    return pd.DataFrame(rows)
