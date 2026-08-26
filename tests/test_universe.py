from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cross_sectional_ml.training import LazySequenceDataset
from cross_sectional_ml.universe import (
    complete_history_universe,
    data_pack_from_panel,
    entry_eligible_universe,
    prepare_model_panel,
)


def test_entry_eligibility_and_complete_history() -> None:
    entry = pd.DataFrame(
        {
            "Code": ["000001.SZ", "000002.SZ", "600001.SH"],
            "Date": [20240102] * 3,
            "is_st": [False, True, False],
            "is_limit_up_open": [False, False, True],
            "is_limit_down_open": [False, False, False],
        }
    )
    eligible = entry_eligible_universe(entry, {20240102: 20231229})
    assert eligible.to_dict("records") == [{"Code": "000001", "Date": 20231229}]

    dates = [20240102, 20240103, 20240104]
    features = pd.DataFrame(
        {
            "Code": ["000001"] * 3 + ["000002"] * 2,
            "Date": dates + [20240102, 20240104],
        }
    )
    history = complete_history_universe(features, dates, sequence_length=3)
    assert history.to_dict("records") == [{"Code": "000001", "Date": 20240104}]


def test_prepare_model_panel_recomputes_target() -> None:
    features = pd.DataFrame(
        {
            "Code": ["000001", "000002", "000003"],
            "Date": [20240102] * 3,
            "f1": [1.0, np.nan, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "Code": ["000001", "000002", "000003"],
            "Date": [20240102] * 3,
            "ret": [-1.0, 0.0, 1.0],
        }
    )
    panel = prepare_model_panel(
        features,
        labels,
        ["f1"],
        raw_return_column="ret",
        target_column="target",
    )
    assert panel["f1"].tolist() == [1.0, 0.0, 3.0]
    assert np.allclose(panel["target"], [-1.0, 0.0, 1.0])
    pack = data_pack_from_panel(
        panel,
        ["f1"],
        target_column="target",
        raw_return_column="ret",
    )
    assert pack.x.shape == (3, 1)
    assert pack.y is not None


def test_context_history_and_outcome_independent_prediction_keys() -> None:
    dates = [20240102, 20240103, 20240104]
    codes = ["000001", "000002", "000003"]
    features = pd.DataFrame(
        [{"Code": code, "Date": date, "f1": float(code[-1])} for date in dates for code in codes]
    )
    labels = pd.DataFrame(
        {
            "Code": codes,
            "Date": [20240104] * 3,
            "ret": [-1.0, 1.0, np.nan],
        }
    )
    endpoints = pd.DataFrame({"Code": codes, "Date": [20240104] * 3})
    panel = prepare_model_panel(
        features,
        labels,
        ["f1"],
        raw_return_column="ret",
        target_column="target",
        eligible_keys=endpoints,
        history_keys=endpoints,
    )
    assert len(panel) == 9
    assert panel["is_eligible"].sum() == 3
    assert panel["is_scoreable"].sum() == 2

    pack = data_pack_from_panel(
        panel,
        ["f1"],
        target_column="target",
        raw_return_column="ret",
    )
    target_positions = np.flatnonzero(panel["is_eligible"].to_numpy())
    sequence = LazySequenceDataset(
        pack,
        sequence_length=3,
        calendar_dates=dates,
        target_positions=target_positions,
    )
    assert len(sequence) == 3
    assert set(sequence.symbols) == set(codes)


def test_sequence_rejects_requested_endpoint_without_complete_history() -> None:
    pack = data_pack_from_panel(
        pd.DataFrame(
            {
                "Code": ["000001", "000001"],
                "Date": [20240102, 20240104],
                "f1": [1.0, 2.0],
            }
        ),
        ["f1"],
        target_column=None,
    )
    with pytest.raises(ValueError, match="lack a complete 3-session history"):
        LazySequenceDataset(
            pack,
            sequence_length=3,
            calendar_dates=[20240102, 20240103, 20240104],
            target_positions=[1],
        )
