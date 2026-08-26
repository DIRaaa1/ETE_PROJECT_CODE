from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .io import normalize_panel_keys
from .labels import zscore_target_by_date
from .training import DataPack


def entry_eligible_universe(
    entry_panel: pd.DataFrame,
    entry_to_signal: Mapping[int, int],
    *,
    allowed_prefixes: tuple[str, ...] = ("0", "3", "6"),
) -> pd.DataFrame:
    """Map eligible entry-session stock keys back to their signal dates."""

    required = {"Code", "Date", "is_st", "is_limit_up_open", "is_limit_down_open"}
    missing = required.difference(entry_panel.columns)
    if missing:
        raise ValueError(f"missing eligibility columns: {sorted(missing)}")
    panel = normalize_panel_keys(entry_panel[list(required)])
    for column in ("is_st", "is_limit_up_open", "is_limit_down_open"):
        panel[column] = panel[column].astype("boolean")
    eligible = (
        panel["Code"].str.startswith(allowed_prefixes)
        & ~panel["is_st"].fillna(True)
        & ~panel["is_limit_up_open"].fillna(True)
        & ~panel["is_limit_down_open"].fillna(True)
    )
    signal_date = {int(entry): int(signal) for entry, signal in entry_to_signal.items()}
    panel["Date"] = panel["Date"].map(signal_date)
    if panel["Date"].isna().any():
        raise ValueError("entry dates must have a preceding session in the trading calendar")
    panel["Date"] = panel["Date"].astype("int64")
    return (
        panel.loc[eligible, ["Code", "Date"]]
        .sort_values(["Date", "Code"], kind="stable")
        .reset_index(drop=True)
    )


def complete_history_universe(
    features: pd.DataFrame,
    trading_dates: Sequence[int],
    *,
    sequence_length: int = 10,
) -> pd.DataFrame:
    """Return stock-days with features on every session in the trailing window."""

    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    keys = normalize_panel_keys(features[["Code", "Date"]])
    calendar = np.asarray(sorted(set(map(int, trading_dates))), dtype=np.int64)
    calendar_position = {date: position for position, date in enumerate(calendar.tolist())}
    rows: list[pd.DataFrame] = []
    for _, group in keys.groupby("Code", sort=False):
        ordered = group.sort_values("Date", kind="stable")
        positions = ordered["Date"].map(calendar_position).to_numpy(dtype=float)
        valid = np.zeros(len(ordered), dtype=bool)
        if len(ordered) >= sequence_length:
            for end in range(sequence_length - 1, len(ordered)):
                window = positions[end - sequence_length + 1 : end + 1]
                valid[end] = np.isfinite(window).all() and np.array_equal(
                    np.diff(window), np.ones(sequence_length - 1)
                )
        rows.append(ordered.loc[valid, ["Code", "Date"]])
    if not rows:
        return pd.DataFrame(columns=["Code", "Date"])
    return (
        pd.concat(rows, ignore_index=True).sort_values(["Date", "Code"], kind="stable").reset_index(drop=True)
    )


def prepare_model_panel(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    raw_return_column: str,
    target_column: str,
    eligible_keys: pd.DataFrame | None = None,
    history_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one feature-history panel with ex-ante eligible target rows marked."""

    names = list(feature_names)
    feature_panel = normalize_panel_keys(features[["Code", "Date", *names]])
    label_panel = normalize_panel_keys(labels[["Code", "Date", raw_return_column]])
    eligible = feature_panel[["Code", "Date"]].copy()
    for keys in (eligible_keys, history_keys):
        if keys is not None:
            selected = normalize_panel_keys(keys[["Code", "Date"]])
            eligible = eligible.merge(selected, on=["Code", "Date"], how="inner", validate="one_to_one")
    eligible["is_eligible"] = True

    panel = feature_panel.merge(eligible, on=["Code", "Date"], how="left", validate="one_to_one")
    panel["is_eligible"] = panel["is_eligible"].astype("boolean").fillna(False).astype(bool)
    label_panel[raw_return_column] = pd.to_numeric(label_panel[raw_return_column], errors="coerce")
    panel = panel.merge(label_panel, on=["Code", "Date"], how="left", validate="one_to_one")
    scoreable = panel["is_eligible"] & np.isfinite(panel[raw_return_column].to_numpy(dtype=float))
    target_rows = zscore_target_by_date(
        panel.loc[scoreable, ["Code", "Date", raw_return_column]].copy(),
        raw_return_column,
        target_column,
    )
    target_rows = target_rows[np.isfinite(target_rows[target_column].to_numpy(dtype=float))]
    panel = panel.merge(
        target_rows[["Code", "Date", target_column]],
        on=["Code", "Date"],
        how="left",
        validate="one_to_one",
    )
    panel["is_scoreable"] = panel["is_eligible"] & np.isfinite(panel[target_column].to_numpy(dtype=float))

    values = panel[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = 0.0
    panel[names] = values
    return (
        panel[
            [
                "Code",
                "Date",
                *names,
                target_column,
                raw_return_column,
                "is_eligible",
                "is_scoreable",
            ]
        ]
        .sort_values(["Date", "Code"], kind="stable")
        .reset_index(drop=True)
    )


def data_pack_from_panel(
    panel: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    target_column: str | None,
    raw_return_column: str | None = None,
) -> DataPack:
    """Convert a prepared panel to the common training and inference contract."""

    names = list(feature_names)
    return DataPack(
        x=panel[names].to_numpy(dtype=np.float32),
        y=None if target_column is None else panel[target_column].to_numpy(dtype=np.float32),
        dates=panel["Date"].to_numpy(dtype=np.int64),
        symbols=panel["Code"].astype(str).to_numpy(),
        feature_names=names,
        raw_returns=(
            None if raw_return_column is None else panel[raw_return_column].to_numpy(dtype=np.float32)
        ),
    )
