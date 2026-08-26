from __future__ import annotations

import numpy as np
import pandas as pd

from cross_sectional_ml.labels import (
    DE_LABEL,
    O2O_LABEL,
    build_de_label,
    build_o2o_label,
    compute_adjusted_vwap,
)


def _constant_vwap_bars(
    values: tuple[float, ...],
    minutes: tuple[int, ...],
    *,
    early_values: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    codes = ("000001.SZ", "000002.SZ", "600000.SH")
    for code, value, early_value in zip(codes, values, early_values or values):
        for minute in minutes:
            minute_value = early_value if minute < 933 else value
            rows.append(
                {
                    "code": code,
                    "minute": minute,
                    "amount": minute_value,
                    "vol": 10.0,
                    "adj_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_adjusted_vwap_uses_amount_factor_and_volume_formula() -> None:
    bars = pd.DataFrame(
        {
            "code": ["000001.SZ", "000001.SZ"],
            "minute": [930, 931],
            "amount": [10.0, 10.0],
            "vol": [10.0, 10.0],
            "adj_factor": [1.0, 2.0],
        }
    )

    result = compute_adjusted_vwap(bars)

    assert result.loc[0, "Code"] == "000001"
    assert result.loc[0, "adjusted_vwap"] == 15.0


def test_minute_filter_accepts_clock_strings() -> None:
    bars = pd.DataFrame(
        {
            "Code": ["000001"] * 3,
            "Minute": ["09:30:00", "09:31:00", "09:32:00"],
            "amount": [10.0, 20.0, 900.0],
            "vol": [10.0, 10.0, 10.0],
        }
    )
    result = compute_adjusted_vwap(bars, minutes=(930, 931))
    assert result.loc[0, "adjusted_vwap"] == 15.0


def test_o2o_label_is_open6_return_and_ddof1_cross_sectional_zscore() -> None:
    entry = _constant_vwap_bars((10.0, 10.0, 10.0), O2O_LABEL.entry_minutes)
    exit_ = _constant_vwap_bars((9.0, 10.0, 11.0), O2O_LABEL.exit_minutes)

    label = build_o2o_label(entry, exit_, signal_date=20240102)

    np.testing.assert_allclose(label[O2O_LABEL.raw_column], [-0.1, 0.0, 0.1], atol=1e-7)
    np.testing.assert_allclose(label[O2O_LABEL.target_column], [-1.0, 0.0, 1.0], atol=1e-6)
    assert list(label.columns) == ["Code", "Date", O2O_LABEL.raw_column, O2O_LABEL.target_column]


def test_de_entry_ignores_0930_to_0932_and_uses_0933_to_0935() -> None:
    all_entry_minutes = (930, 931, 932, 933, 934, 935)
    entry = _constant_vwap_bars(
        (10.0, 10.0, 10.0),
        all_entry_minutes,
        early_values=(100.0, 200.0, 300.0),
    )
    exit_ = _constant_vwap_bars((9.0, 10.0, 11.0), DE_LABEL.exit_minutes)

    label = build_de_label(entry, exit_, signal_date="2024-01-02")

    np.testing.assert_allclose(label[DE_LABEL.raw_column], [-0.1, 0.0, 0.1], atol=1e-7)
    np.testing.assert_allclose(label[DE_LABEL.target_column], [-1.0, 0.0, 1.0], atol=1e-6)
