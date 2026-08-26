"""Minimal reproducible backtest used by the dissertation's final tables."""

from .ensemble import average_daily_spearman, rank_zscore_ensemble
from .metrics import correlation, daily_predictive_metrics
from .portfolio import (
    MODES,
    QUANTILES,
    backtest_daily,
    backtest_grid,
    portfolio_day,
    select_day,
    summarise_daily,
)
from .predictions import (
    EIGHT_MODELS,
    align_eight_model_predictions,
    attach_outcomes,
    read_model_predictions,
    read_prediction,
)
from .reporting import portfolio_summary, predictive_summary
from .rolling import combine_rolling_years

__all__ = [
    "EIGHT_MODELS",
    "MODES",
    "QUANTILES",
    "align_eight_model_predictions",
    "attach_outcomes",
    "average_daily_spearman",
    "backtest_daily",
    "backtest_grid",
    "combine_rolling_years",
    "correlation",
    "daily_predictive_metrics",
    "portfolio_day",
    "portfolio_summary",
    "predictive_summary",
    "rank_zscore_ensemble",
    "read_model_predictions",
    "read_prediction",
    "select_day",
    "summarise_daily",
]
