"""O2O and delayed-entry VWAP return labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .io import normalize_code_series, normalize_date_series, normalize_minute_series

Protocol = Literal["o2o", "de"]


@dataclass(frozen=True)
class LabelSpec:
    protocol: Protocol
    entry_minutes: tuple[int, ...]
    exit_minutes: tuple[int, ...]
    raw_column: str
    target_column: str


O2O_LABEL = LabelSpec(
    protocol="o2o",
    entry_minutes=(930, 931, 932, 933, 934, 935),
    exit_minutes=(930, 931, 932, 933, 934, 935),
    raw_column="oto_d1_open6_to_open6",
    target_column="oto_d1_open6_to_open6_zscore",
)
DE_LABEL = LabelSpec(
    protocol="de",
    entry_minutes=(933, 934, 935),
    exit_minutes=(930, 931, 932, 933, 934, 935),
    raw_column="oto_gap_d1_open3_to_open6",
    target_column="oto_gap_d1_open3_to_open6_zscore",
)
LABEL_SPECS: dict[str, LabelSpec] = {"o2o": O2O_LABEL, "de": DE_LABEL}


def get_label_spec(protocol: str) -> LabelSpec:
    """Return the frozen protocol label contract."""

    key = protocol.strip().lower()
    try:
        return LABEL_SPECS[key]
    except KeyError as exc:
        raise ValueError("protocol must be 'o2o' or 'de'") from exc


def compute_adjusted_vwap(
    bars: pd.DataFrame,
    *,
    minutes: tuple[int, ...] | None = None,
    output_column: str = "adjusted_vwap",
) -> pd.DataFrame:
    """Compute ``10 * sum(amount * adj_factor) / sum(vol)`` by security."""

    code_source = "Code" if "Code" in bars.columns else "code"
    if code_source not in bars.columns:
        raise ValueError("bars have neither 'Code' nor 'code'")
    missing = sorted({"amount", "vol"}.difference(bars.columns))
    if missing:
        raise ValueError(f"bars are missing required columns: {missing}")

    frame = bars.copy()
    frame["Code"] = normalize_code_series(frame[code_source])
    if frame["Code"].isna().any():
        raise ValueError("bars contain invalid security codes")
    if minutes is not None:
        minute_source = "Minute" if "Minute" in frame.columns else "minute"
        if minute_source not in frame.columns:
            raise ValueError("minute filter requested but bars have no minute column")
        frame["_minute"] = normalize_minute_series(frame[minute_source])
        frame = frame[frame["_minute"].isin(minutes)].copy()
    if frame.empty:
        return pd.DataFrame({"Code": pd.Series(dtype="string"), output_column: pd.Series(dtype="float64")})

    amount = pd.to_numeric(frame["amount"], errors="coerce")
    volume = pd.to_numeric(frame["vol"], errors="coerce")
    if "adj_factor" in frame.columns:
        factor = pd.to_numeric(frame["adj_factor"], errors="coerce").fillna(1.0)
    else:
        factor = pd.Series(1.0, index=frame.index)
    frame["_adjusted_amount"] = amount * factor
    frame["_volume"] = volume
    aggregate = frame.groupby("Code", sort=True).agg(
        adjusted_amount=("_adjusted_amount", lambda values: values.sum(min_count=1)),
        volume=("_volume", lambda values: values.sum(min_count=1)),
    )
    numerator = 10.0 * aggregate["adjusted_amount"].to_numpy(dtype="float64")
    denominator = aggregate["volume"].to_numpy(dtype="float64")
    values = np.full(len(aggregate), np.nan, dtype="float64")
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    values[valid] = numerator[valid] / denominator[valid]
    aggregate[output_column] = values
    return aggregate[[output_column]].reset_index()


def zscore_target_by_date(
    frame: pd.DataFrame,
    raw_column: str,
    target_column: str,
    *,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Apply an ordinary within-date sample z-score (``ddof=1``)."""

    missing = sorted({date_column, raw_column}.difference(frame.columns))
    if missing:
        raise ValueError(f"target frame is missing required columns: {missing}")
    output = frame.copy()
    output[raw_column] = pd.to_numeric(output[raw_column], errors="coerce")
    zscore = pd.Series(np.nan, index=output.index, dtype="float64")
    for _, group in output.groupby(date_column, sort=False):
        values = group[raw_column]
        finite = np.isfinite(values.to_numpy(dtype="float64"))
        if finite.sum() < 2:
            continue
        finite_values = values.iloc[np.flatnonzero(finite)].to_numpy(dtype="float64")
        standard_deviation = finite_values.std(ddof=1)
        if not np.isfinite(standard_deviation) or standard_deviation <= 0:
            continue
        mean = finite_values.mean()
        zscore.loc[group.index[finite]] = (finite_values - mean) / standard_deviation
    output[target_column] = zscore.astype("float32")
    return output


def build_vwap_label(
    entry_bars: pd.DataFrame,
    exit_bars: pd.DataFrame,
    *,
    protocol: str,
    signal_date: object,
) -> pd.DataFrame:
    """Build one signal-date protocol label from entry- and exit-session bars."""

    spec = get_label_spec(protocol)
    normalized_date = normalize_date_series(pd.Series([signal_date])).iloc[0]
    if pd.isna(normalized_date):
        raise ValueError(f"invalid signal date: {signal_date!r}")
    entry = compute_adjusted_vwap(
        entry_bars,
        minutes=spec.entry_minutes,
        output_column="entry_adjusted_vwap",
    )
    exit_ = compute_adjusted_vwap(
        exit_bars,
        minutes=spec.exit_minutes,
        output_column="exit_adjusted_vwap",
    )
    label = entry.merge(exit_, on="Code", how="inner", validate="one_to_one")
    entry_values = label["entry_adjusted_vwap"].to_numpy(dtype="float64")
    exit_values = label["exit_adjusted_vwap"].to_numpy(dtype="float64")
    raw_return = np.full(len(label), np.nan, dtype="float64")
    valid = np.isfinite(entry_values) & np.isfinite(exit_values) & (np.abs(entry_values) > 1e-12)
    raw_return[valid] = exit_values[valid] / entry_values[valid] - 1.0
    label.insert(1, "Date", int(normalized_date))
    label[spec.raw_column] = raw_return.astype("float32")
    label = zscore_target_by_date(label, spec.raw_column, spec.target_column)
    return (
        label[["Code", "Date", spec.raw_column, spec.target_column]]
        .sort_values("Code", kind="stable")
        .reset_index(drop=True)
    )


def build_o2o_label(
    entry_bars: pd.DataFrame,
    exit_bars: pd.DataFrame,
    *,
    signal_date: object,
) -> pd.DataFrame:
    """Build the t+1 open6 to t+2 open6 label."""

    return build_vwap_label(entry_bars, exit_bars, protocol="o2o", signal_date=signal_date)


def build_de_label(
    entry_bars: pd.DataFrame,
    exit_bars: pd.DataFrame,
    *,
    signal_date: object,
) -> pd.DataFrame:
    """Build the t+1 09:33--09:35 to t+2 open6 label."""

    return build_vwap_label(entry_bars, exit_bars, protocol="de", signal_date=signal_date)
