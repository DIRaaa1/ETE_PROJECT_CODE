from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path("src")))

from cross_sectional_ml.backtest import (
    EIGHT_MODELS,
    align_eight_model_predictions,
    attach_outcomes,
    average_daily_spearman,
    backtest_daily,
    combine_rolling_years,
    rank_zscore_ensemble,
    read_prediction,
    select_day,
)


def _prediction_sources(keys: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sources = {}
    for offset, model in enumerate(EIGHT_MODELS):
        frame = keys.copy()
        frame["prediction"] = np.arange(len(frame), dtype=float) + offset / 100.0
        sources[model] = frame
    return sources


def test_prediction_read_alignment_and_outcome_left_join(tmp_path: Path) -> None:
    keys = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "Code": ["000002", "000001", "000001"],
        }
    )
    one_path = tmp_path / "prediction.csv"
    _prediction_sources(keys)["lgb"].to_csv(one_path, index=False)
    one = read_prediction(one_path)
    assert one["Code"].tolist() == ["000001", "000002", "000001"]

    sources = _prediction_sources(keys)
    sources["lgb"] = one_path
    wide = align_eight_model_predictions(sources)
    assert wide.columns.tolist() == ["Date", "Code", *EIGHT_MODELS]
    assert wide[["Date", "Code"]].equals(one[["Date", "Code"]])

    bad = _prediction_sources(keys)
    bad["mamba"] = bad["mamba"].iloc[:-1]
    with pytest.raises(ValueError, match="keys do not match"):
        align_eight_model_predictions(bad)

    outcomes = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Code": ["000001", "000001"],
            "return": [0.1, 0.2],
        }
    )
    attached = attach_outcomes(wide, outcomes)
    assert len(attached) == len(wide)
    assert attached["return"].isna().sum() == 1


def test_integer_dates_and_codes_are_normalised() -> None:
    frame = pd.DataFrame({"Date": [20240102, 20240102], "Code": [1, 2], "prediction": [0.1, 0.2]})
    result = read_prediction(frame)
    assert result["Date"].dt.strftime("%Y%m%d").tolist() == ["20240102", "20240102"]
    assert result["Code"].tolist() == ["000001", "000002"]


@pytest.mark.parametrize(
    ("quantile", "expected_count"),
    [(1.0, 5), (0.75, 4), (0.50, 3), (0.25, 2)],
)
def test_cumulative_quantiles_use_ceil(quantile: float, expected_count: int) -> None:
    day = pd.DataFrame(
        {
            "Code": ["E", "D", "C", "B", "A"],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    selected = select_day(day, mode="signed", quantile=quantile)
    assert len(selected) == expected_count


def test_signed_long_only_tie_break_and_native_missing_accounting() -> None:
    panel = pd.DataFrame(
        {
            "Date": ["2024-01-02"] * 4,
            "Code": ["B", "A", "C", "D"],
            "score": [2.0, 2.0, -3.0, 0.0],
            "return": [5.0, np.nan, -1.0, 999.0],
        }
    )

    signed_selected = select_day(panel, mode="signed", quantile=0.50)
    assert signed_selected["Code"].tolist() == ["C", "A"]
    signed = backtest_daily(panel, mode="signed", quantile=0.50).iloc[0]
    assert signed["pnl"] == pytest.approx(1.0)
    assert signed["notional"] == 2.0
    assert signed["ppd"] == pytest.approx(0.5)
    assert signed["missing_return_count"] == 1
    assert signed["hit_ratio"] == 1.0
    assert signed["long_share"] == 0.5

    long_selected = select_day(panel, mode="long_only", quantile=0.50)
    assert long_selected["Code"].tolist() == ["A"]
    long_only = backtest_daily(panel, mode="long_only", quantile=0.50).iloc[0]
    assert long_only["pnl"] == 0.0
    assert long_only["notional"] == 1.0
    assert long_only["missing_return_count"] == 1
    assert long_only["hit_ratio"] == 0.0
    assert long_only["long_share"] == 1.0


def test_rolling_years_concatenate_and_recompute_summary() -> None:
    panel = pd.DataFrame(
        {
            "Date": [
                "2024-01-02",
                "2024-01-03",
                "2025-01-02",
                "2025-01-02",
                "2025-01-03",
                "2025-01-03",
            ],
            "Code": ["A", "A", "A", "B", "A", "B"],
            "score": [1.0] * 6,
            "return": [1.0, 3.0, -1.0, -1.0, 1.0, 0.0],
        }
    )
    daily_2024 = backtest_daily(
        panel[pd.to_datetime(panel["Date"]).dt.year.eq(2024)],
        mode="long_only",
        quantile=1.0,
    )
    daily_2025 = backtest_daily(
        panel[pd.to_datetime(panel["Date"]).dt.year.eq(2025)],
        mode="long_only",
        quantile=1.0,
    )
    combined, summary = combine_rolling_years(daily_2024, daily_2025)

    assert combined["pnl"].tolist() == [1.0, 3.0, -2.0, 1.0]
    assert summary["pnl"] == 3.0
    assert summary["notional"] == 6.0
    assert summary["ppd"] == pytest.approx(0.5)
    assert summary["hit_ratio"] == pytest.approx(3.0 / 5.0)
    assert summary["long_share"] == 1.0
    expected_sharpe = np.sqrt(252.0) * np.mean([1.0, 3.0, -2.0, 1.0]) / np.std([1.0, 3.0, -2.0, 1.0], ddof=1)
    assert summary["sharpe_ratio"] == pytest.approx(expected_sharpe)


def test_daily_spearman_is_equal_date_mean_and_ensemble_is_rank_zscore() -> None:
    rows: list[dict[str, object]] = []
    for date, count in [("2024-01-02", 3), ("2025-01-02", 5)]:
        for index in range(count):
            ascending = float(index + 1)
            if date.startswith("2024") and index < 2:
                ascending = 1.0
            row: dict[str, object] = {
                "Date": date,
                "Code": f"{index:06d}",
                **{model: ascending for model in EIGHT_MODELS},
            }
            if date.startswith("2025"):
                row["xgb"] = float(count - index)
            rows.append(row)
    panel = pd.DataFrame(rows)

    correlation = average_daily_spearman(panel)
    assert correlation.loc["lgb", "xgb"] == pytest.approx(0.0, abs=1e-12)
    assert correlation.loc["lgb", "mlp"] == pytest.approx(1.0)

    ensemble = rank_zscore_ensemble(panel)
    first_day = ensemble[ensemble["Date"].dt.year.eq(2024)].sort_values("Code")
    average_tie_ranks = pd.Series([1.5, 1.5, 3.0])
    expected_first = (average_tie_ranks - average_tie_ranks.mean()) / average_tie_ranks.std(ddof=1)
    assert first_day["ensemble_score"].to_numpy() == pytest.approx(expected_first.to_numpy())

    second_day = ensemble[ensemble["Date"].dt.year.eq(2025)].sort_values("Code")
    base_ranks = pd.Series(np.arange(1.0, 6.0))
    base_z = (base_ranks - base_ranks.mean()) / base_ranks.std(ddof=1)
    # Seven ascending models and one reversed model: (7z - z) / 8 = 0.75z.
    assert second_day["ensemble_score"].to_numpy() == pytest.approx(0.75 * base_z.to_numpy())
