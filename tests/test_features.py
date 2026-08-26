from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cross_sectional_ml.features import (
    D1_FEATURE_COLUMNS,
    DE_RAW_FEATURE_COLUMNS,
    EO_FEATURE_COLUMNS,
    M30_FEATURE_COLUMNS,
    O2O_RAW_FEATURE_COLUMNS,
    build_de_raw_features,
    build_o2o_raw_features,
    legacy_mad3_v1,
    load_feature_manifest,
    project_feature_panel,
    standardize_feature_frame_legacy_v1,
    validate_manifest_pair,
)
from cross_sectional_ml.io import read_parquet, write_parquet


def _daily_history() -> tuple[pd.DataFrame, int]:
    dates = pd.bdate_range("2024-01-02", periods=65)
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(("000001.SZ", "600000.SH")):
        previous_close = 10.0 + code_index
        for day_index, date in enumerate(dates):
            close = 10.0 + code_index + day_index * 0.01
            rows.append(
                {
                    "code": code,
                    "date": date,
                    "open": close - 0.02,
                    "high": close + 0.05,
                    "low": close - 0.05,
                    "close": close,
                    "pre_close": previous_close,
                    "vol": 1_000.0 + day_index,
                    "amount": close * (1_000.0 + day_index) / 10.0,
                    "adj_factor": 1.0,
                    "vwap": close,
                }
            )
            previous_close = close
    return pd.DataFrame(rows), int(dates[-1].strftime("%Y%m%d"))


def _minute_panel(minutes: tuple[int, ...], *, price_shift: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code_index, code in enumerate(("000001.SZ", "600000.SH")):
        for minute_index, minute in enumerate(minutes):
            price = 10.5 + code_index + price_shift + minute_index * 0.001
            volume = 100.0 + minute_index
            rows.append(
                {
                    "code": code,
                    "minute": minute,
                    "open": price - 0.01,
                    "high": price + 0.02,
                    "low": price - 0.02,
                    "close": price,
                    "vol": volume,
                    "amount": price * volume,
                    "adj_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_builders_return_frozen_599_and_657_contracts() -> None:
    daily, signal_date = _daily_history()
    signal_minutes = _minute_panel(
        (930, 1000, 1001, 1030, 1031, 1100, 1101, 1130, 1301, 1330, 1331, 1400, 1401, 1430, 1431, 1500)
    )
    early_open = _minute_panel((930, 931), price_shift=0.1)

    o2o = build_o2o_raw_features(daily, signal_minutes, signal_date=signal_date)
    de = build_de_raw_features(daily, signal_minutes, early_open, signal_date=signal_date)

    assert len(M30_FEATURE_COLUMNS) == 560
    assert len(D1_FEATURE_COLUMNS) == 39
    assert len(EO_FEATURE_COLUMNS) == 58
    assert len(O2O_RAW_FEATURE_COLUMNS) == 599
    assert len(DE_RAW_FEATURE_COLUMNS) == 657
    assert tuple(o2o.columns) == ("Code", "Date", *O2O_RAW_FEATURE_COLUMNS)
    assert tuple(de.columns) == ("Code", "Date", *DE_RAW_FEATURE_COLUMNS)
    assert len(o2o) == len(de) == 2
    assert all(dtype == np.dtype("float32") for dtype in o2o.iloc[:, 2:].dtypes)
    assert all(dtype == np.dtype("float32") for dtype in de.iloc[:, 2:].dtypes)


def test_legacy_mad3_uses_total_row_denominator_and_zero_fills_missing() -> None:
    values = np.array([[1.0], [2.0], [3.0], [np.nan]], dtype="float32")
    result = legacy_mad3_v1(values, min_valid_count=2)

    expected_scale = np.sqrt(2.0 / 3.0)  # denominator is n_rows - 1, not finite_count - 1
    np.testing.assert_allclose(result[:3, 0], [-1.0 / expected_scale, 0.0, 1.0 / expected_scale])
    assert result[3, 0] == 0.0
    assert result.dtype == np.float32


def test_packaged_manifests_are_ordered_and_project_exactly() -> None:
    o2o = load_feature_manifest("o2o")
    de = load_feature_manifest("de")
    validate_manifest_pair(o2o, de)
    source = pd.DataFrame(
        {
            "Date": [20240102],
            "Code": ["000001"],
            "unregistered": [99.0],
            **{column: [float(index)] for index, column in enumerate(reversed(de.columns))},
        }
    )

    projected = project_feature_panel(source, de)

    assert tuple(projected.columns) == ("Code", "Date", *de.columns)
    assert len(projected.columns) == 604
    assert projected.loc[0, de.columns[0]] == float(len(de.columns) - 1)


def test_projection_rejects_missing_columns() -> None:
    manifest = load_feature_manifest("o2o")
    with pytest.raises(ValueError, match="missing"):
        project_feature_panel(pd.DataFrame({"Code": ["000001"], "Date": [20240102]}), manifest)


def test_feature_normalization_rejects_multiple_dates() -> None:
    frame = pd.DataFrame({"Code": ["000001", "000001"], "Date": [20240102, 20240103], "M30_X": [1.0, 2.0]})
    with pytest.raises(ValueError, match="one signal date"):
        standardize_feature_frame_legacy_v1(frame, ["M30_X"], min_valid_count=1)


def test_simple_parquet_projection_round_trip(tmp_path) -> None:
    frame = pd.DataFrame({"Code": ["000001"], "Date": [20240102], "factor": [1.25]})
    path = write_parquet(frame, tmp_path / "20240102.parquet")
    restored = read_parquet(path, columns=["Date", "factor"])
    pd.testing.assert_frame_equal(restored, frame[["Date", "factor"]])
