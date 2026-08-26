"""Pure DataFrame builders for the frozen 599/657 raw feature panels.

The public builder consumes one historical daily panel, the signal-day minute
panel, and (for DE) the next-session 09:30/09:31 panel.  It intentionally has
no filesystem or multiprocessing policy.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..io import normalize_code_series, normalize_date_series, normalize_minute_series
from .definitions import (
    D1_FEATURE_COLUMNS,
    DE_RAW_FEATURE_COLUMNS,
    EO_FEATURE_COLUMNS,
    M30_FEATURE_COLUMNS,
    M30_FIELDS,
    M30_SLOTS,
    O2O_RAW_FEATURE_COLUMNS,
)

EPS = 1e-12
DEFAULT_CODE_PREFIXES = ("0", "3", "6")


def safe_div(numerator: object, denominator: object, *, eps: float = EPS) -> np.ndarray:
    """Element-wise finite division, returning NaN for invalid denominators."""

    a = np.asarray(numerator, dtype="float64")
    b = np.asarray(denominator, dtype="float64")
    shape = np.broadcast_shapes(a.shape, b.shape)
    aa = np.broadcast_to(a, shape)
    bb = np.broadcast_to(b, shape)
    out = np.full(shape, np.nan, dtype="float64")
    valid = np.isfinite(aa) & np.isfinite(bb) & (np.abs(bb) > eps)
    out[valid] = aa[valid] / bb[valid]
    return out


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _filter_codes(frame: pd.DataFrame, prefixes: tuple[str, ...]) -> pd.DataFrame:
    out = frame.copy()
    source = "Code" if "Code" in out.columns else "code"
    if source not in out.columns:
        raise ValueError("input has neither 'Code' nor 'code'")
    out["Code"] = normalize_code_series(out[source])
    return out[out["Code"].str.startswith(prefixes, na=False)].copy()


def prepare_daily_panel(
    raw_daily: pd.DataFrame,
    *,
    code_prefixes: tuple[str, ...] = DEFAULT_CODE_PREFIXES,
) -> pd.DataFrame:
    """Normalize raw daily bars to the adjusted contract used by D1/M30.

    Expected units match the archived source: daily ``vol`` is in lots and
    ``amount`` in thousands of currency units.  Prices and amount are adjusted
    with ``adj_factor``; a missing factor defaults to one.
    """

    _require_columns(
        raw_daily,
        ("open", "high", "low", "close", "pre_close", "vol", "amount"),
        name="daily panel",
    )
    out = _filter_codes(raw_daily, code_prefixes)
    date_source = "Date" if "Date" in out.columns else "date"
    if date_source not in out.columns:
        raise ValueError("daily panel has neither 'Date' nor 'date'")
    out["Date"] = normalize_date_series(out[date_source])
    if out["Date"].isna().any():
        raise ValueError("daily panel contains invalid dates")

    numeric = ["open", "high", "low", "close", "pre_close", "vol", "amount"]
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "adj_factor" not in out.columns:
        out["adj_factor"] = 1.0
    factor = pd.to_numeric(out["adj_factor"], errors="coerce").fillna(1.0)
    if "vwap" in out.columns:
        vwap = pd.to_numeric(out["vwap"], errors="coerce")
    else:
        vwap = pd.Series(np.nan, index=out.index, dtype="float64")
    if vwap.isna().all():
        vwap = pd.Series(safe_div(out["amount"] * 1000.0, out["vol"] * 100.0), index=out.index)

    panel = pd.DataFrame(
        {
            "Code": out["Code"].array,
            "Date": out["Date"].astype("int64").array,
            "open_adj": out["open"].to_numpy() * factor.to_numpy(),
            "high_adj": out["high"].to_numpy() * factor.to_numpy(),
            "low_adj": out["low"].to_numpy() * factor.to_numpy(),
            "close_adj": out["close"].to_numpy() * factor.to_numpy(),
            "pre_close_adj": out["pre_close"].to_numpy() * factor.to_numpy(),
            "vwap_adj": vwap.to_numpy() * factor.to_numpy(),
            "volume_shares": out["vol"].to_numpy() * 100.0,
            "amount_adj": out["amount"].to_numpy() * 1000.0 * factor.to_numpy(),
            "adj_factor": factor.to_numpy(),
        }
    )
    if panel.duplicated(["Code", "Date"]).any():
        raise ValueError("daily panel contains duplicate (Code, Date) keys")
    return panel.sort_values(["Code", "Date"], kind="stable").reset_index(drop=True)


def prepare_minute_panel(
    raw_minutes: pd.DataFrame,
    *,
    code_prefixes: tuple[str, ...] = DEFAULT_CODE_PREFIXES,
) -> pd.DataFrame:
    """Normalize one trading session of minute bars.

    Minute prices are already back-adjusted in the archived HFQ source.  Only
    minute amount is multiplied by ``adj_factor``, matching that implementation.
    """

    _require_columns(raw_minutes, ("open", "high", "low", "close", "vol", "amount"), name="minute panel")
    out = _filter_codes(raw_minutes, code_prefixes)
    minute_source = "Minute" if "Minute" in out.columns else "minute"
    if minute_source not in out.columns:
        raise ValueError("minute panel has neither 'Minute' nor 'minute'")
    out["Minute"] = normalize_minute_series(out[minute_source])
    if out["Minute"].isna().any():
        raise ValueError("minute panel contains invalid minute values")
    for column in ("open", "high", "low", "close", "vol", "amount"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    if "adj_factor" not in out.columns:
        out["adj_factor"] = 1.0
    factor = pd.to_numeric(out["adj_factor"], errors="coerce").fillna(1.0)
    out["amount_adj"] = out["amount"] * factor
    if out.duplicated(["Code", "Minute"]).any():
        raise ValueError("minute panel contains duplicate (Code, Minute) keys")
    columns = ["Code", "Minute", "open", "high", "low", "close", "vol", "amount_adj"]
    return out[columns].sort_values(["Code", "Minute"], kind="stable").reset_index(drop=True)


def build_daily_features(adjusted_daily: pd.DataFrame) -> pd.DataFrame:
    """Build the 39 D1 features from a normalized historical daily panel."""

    _require_columns(
        adjusted_daily,
        (
            "Code",
            "Date",
            "open_adj",
            "high_adj",
            "low_adj",
            "close_adj",
            "vwap_adj",
            "volume_shares",
            "amount_adj",
        ),
        name="adjusted daily panel",
    )
    if adjusted_daily.empty:
        return pd.DataFrame(columns=("Code", "Date", *D1_FEATURE_COLUMNS))
    df = adjusted_daily.sort_values(["Code", "Date"], kind="stable").reset_index(drop=True).copy()
    grouped = df.groupby("Code", sort=False)
    previous_close = grouped["close_adj"].shift(1)
    previous_vwap = grouped["vwap_adj"].shift(1)
    previous_volume = grouped["volume_shares"].shift(1)
    previous_amount = grouped["amount_adj"].shift(1)
    high_low = df["high_adj"] - df["low_adj"]
    log_amount = np.log1p(df["amount_adj"])
    return_open_close = safe_div(df["close_adj"], df["open_adj"]) - 1.0
    return_close_close = safe_div(df["close_adj"], previous_close) - 1.0
    range_open = safe_div(high_low, df["open_adj"])

    features = pd.DataFrame({"Code": df["Code"].array, "Date": df["Date"].array})
    features["D1_OPEN_REL_CLOSE"] = safe_div(df["open_adj"], df["close_adj"])
    features["D1_HIGH_REL_CLOSE"] = safe_div(df["high_adj"], df["close_adj"])
    features["D1_LOW_REL_CLOSE"] = safe_div(df["low_adj"], df["close_adj"])
    features["D1_VWAP_REL_CLOSE"] = safe_div(df["vwap_adj"], df["close_adj"])
    features["D1_RANGE_OPEN"] = range_open
    features["D1_BODY"] = safe_div(df["close_adj"] - df["open_adj"], df["open_adj"])
    features["D1_ABS_BODY"] = safe_div(np.abs(df["close_adj"] - df["open_adj"]), df["open_adj"])
    features["D1_UPPER_SHADOW"] = safe_div(
        np.maximum(df["high_adj"] - np.maximum(df["open_adj"], df["close_adj"]), 0.0),
        df["open_adj"],
    )
    features["D1_LOWER_SHADOW"] = safe_div(
        np.maximum(np.minimum(df["open_adj"], df["close_adj"]) - df["low_adj"], 0.0),
        df["open_adj"],
    )
    features["D1_BODY_RANGE_RATIO"] = safe_div(np.abs(df["close_adj"] - df["open_adj"]), high_low)
    features["D1_CLOSE_POS"] = safe_div(df["close_adj"] - df["low_adj"], high_low)
    features["D1_VWAP_POS"] = safe_div(df["vwap_adj"] - df["low_adj"], high_low)
    features["D1_RET_OC"] = return_open_close
    features["D1_RET_CC_1D"] = return_close_close
    features["D1_GAP"] = safe_div(df["open_adj"], previous_close) - 1.0
    features["D1_RET_VWAP_1D"] = safe_div(df["vwap_adj"], previous_vwap) - 1.0
    features["D1_CLOSE_VWAP"] = safe_div(df["close_adj"], df["vwap_adj"]) - 1.0
    features["D1_OPEN_VWAP"] = safe_div(df["open_adj"], df["vwap_adj"]) - 1.0
    features["D1_LOG_VOLUME"] = np.log1p(df["volume_shares"])
    features["D1_LOG_AMOUNT"] = log_amount
    features["D1_VOL_CHG_1D"] = safe_div(df["volume_shares"], previous_volume) - 1.0
    features["D1_AMT_CHG_1D"] = safe_div(df["amount_adj"], previous_amount) - 1.0
    features["D1_VWAP_AMOUNT_INTENSITY"] = safe_div(
        df["amount_adj"], np.abs(df["close_adj"] - df["open_adj"])
    )
    features["D1_RET_PER_AMOUNT"] = safe_div(return_open_close, log_amount)
    features["D1_RANGE_PER_AMOUNT"] = safe_div(range_open, log_amount)
    for window in (5, 10, 20, 60):
        features[f"D1_RET_{window}D"] = safe_div(df["close_adj"], grouped["close_adj"].shift(window)) - 1.0
    features["D1_REV_1D"] = -return_close_close
    for window in (5, 20, 60):
        moving_average = grouped["close_adj"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).mean()
        )
        features[f"D1_MA_DEV_{window}D"] = safe_div(df["close_adj"], moving_average) - 1.0
    for window in (20, 60):
        low = grouped["low_adj"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).min()
        )
        high = grouped["high_adj"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).max()
        )
        features[f"D1_CLOSE_POS_{window}D"] = safe_div(df["close_adj"] - low, high - low)
        features[f"D1_DIST_HIGH_{window}D"] = safe_div(df["close_adj"], high) - 1.0
        features[f"D1_DIST_LOW_{window}D"] = safe_div(df["close_adj"], low) - 1.0
    return features[["Code", "Date", *D1_FEATURE_COLUMNS]]


def _minute_range(start: int, end: int) -> tuple[int, ...]:
    start_minutes = start // 100 * 60 + start % 100
    end_minutes = end // 100 * 60 + end % 100
    return tuple((value // 60) * 100 + value % 60 for value in range(start_minutes, end_minutes + 1))


_MINUTE_TO_SLOT = {
    minute: slot
    for slot, (start, end) in enumerate(M30_SLOTS, start=1)
    for minute in _minute_range(start, end)
}


def _correlation_by_code(frame: pd.DataFrame, x_column: str, y_column: str, name: str) -> pd.DataFrame:
    sub = frame[["Code", x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if sub.empty:
        return pd.DataFrame({"Code": pd.Series(dtype="string"), name: pd.Series(dtype="float64")})
    sub["_xy"] = sub[x_column] * sub[y_column]
    sub["_x2"] = sub[x_column] ** 2
    sub["_y2"] = sub[y_column] ** 2
    aggregate = sub.groupby("Code", sort=False).agg(
        n=(x_column, "size"),
        sx=(x_column, "sum"),
        sy=(y_column, "sum"),
        sxy=("_xy", "sum"),
        sx2=("_x2", "sum"),
        sy2=("_y2", "sum"),
    )
    numerator = aggregate["n"] * aggregate["sxy"] - aggregate["sx"] * aggregate["sy"]
    denominator_x = aggregate["n"] * aggregate["sx2"] - aggregate["sx"] ** 2
    denominator_y = aggregate["n"] * aggregate["sy2"] - aggregate["sy"] ** 2
    denominator = np.sqrt(np.where(denominator_x * denominator_y > 0, denominator_x * denominator_y, np.nan))
    aggregate[name] = safe_div(numerator, denominator)
    aggregate.loc[aggregate["n"] < 2, name] = np.nan
    return aggregate[[name]].reset_index()


def _aggregate_m30_slot(minutes: pd.DataFrame, slot: int) -> pd.DataFrame:
    frame = minutes[minutes["slot"].eq(slot)].copy()
    if frame.empty:
        return pd.DataFrame()
    frame = frame.sort_values(["Code", "Minute"], kind="stable")
    frame["ret"] = frame.groupby("Code", sort=False)["close"].pct_change(fill_method=None)
    frame["absret"] = frame["ret"].abs()
    frame["ret2"] = frame["ret"] ** 2
    frame["up_vol"] = np.where(frame["ret"] > 0, frame["vol"], 0.0)
    frame["down_vol"] = np.where(frame["ret"] < 0, frame["vol"], 0.0)
    frame["up_amt"] = np.where(frame["ret"] > 0, frame["amount_adj"], 0.0)
    frame["down_amt"] = np.where(frame["ret"] < 0, frame["amount_adj"], 0.0)
    grouped = frame.groupby("Code", sort=False)
    aggregate = grouped.agg(
        OPEN_RAW=("open", "first"),
        CLOSE_RAW=("close", "last"),
        HIGH_RAW=("high", "max"),
        LOW_RAW=("low", "min"),
        VOLUME_RAW=("vol", "sum"),
        AMOUNT_RAW=("amount_adj", "sum"),
        VOLUME_FIRST_RAW=("vol", "first"),
        VOLUME_LAST_RAW=("vol", "last"),
        VOLUME_MEAN_RAW=("vol", "mean"),
        AMOUNT_FIRST_RAW=("amount_adj", "first"),
        AMOUNT_LAST_RAW=("amount_adj", "last"),
        AMOUNT_MEAN_RAW=("amount_adj", "mean"),
        RET_N=("ret", "count"),
        MIN_RET_STD=("ret", "std"),
        RET2_SUM=("ret2", "sum"),
        ABS_MIN_RET_MEAN=("absret", "mean"),
        MAX_MIN_RET=("ret", "max"),
        MIN_MIN_RET=("ret", "min"),
        UP_N=("ret", lambda values: float((values > 0).sum())),
        DOWN_N=("ret", lambda values: float((values < 0).sum())),
        FLAT_N=("ret", lambda values: float((values == 0).sum())),
        UP_VOL=("up_vol", "sum"),
        DOWN_VOL=("down_vol", "sum"),
        UP_AMT=("up_amt", "sum"),
        DOWN_AMT=("down_amt", "sum"),
    ).reset_index()
    aggregate["VWAP_RAW"] = safe_div(aggregate["AMOUNT_RAW"], aggregate["VOLUME_RAW"])
    detail = frame.merge(aggregate[["Code", "VWAP_RAW", "OPEN_RAW"]], on="Code", how="left")
    detail["dev_vwap"] = safe_div(detail["close"], detail["VWAP_RAW"]) - 1.0
    detail["dev_open"] = safe_div(detail["close"], detail["OPEN_RAW"]) - 1.0
    stats = (
        detail.groupby("Code", sort=False)
        .agg(
            PRICE_DEV_VWAP_MEAN=("dev_vwap", "mean"),
            PRICE_DEV_VWAP_ABS_MEAN=("dev_vwap", lambda values: values.abs().mean()),
            PRICE_DEV_VWAP_STD=("dev_vwap", "std"),
            PRICE_MAX_DEV_OPEN=("dev_open", "max"),
            PRICE_MIN_DEV_OPEN=("dev_open", "min"),
        )
        .reset_index()
    )
    aggregate = aggregate.merge(stats, on="Code", how="left")
    correlation_pairs = (
        ("ret", "vol", "RET_VOL_CORR"),
        ("ret", "amount_adj", "RET_AMT_CORR"),
        ("absret", "vol", "ABSRET_VOL_CORR"),
        ("absret", "amount_adj", "ABSRET_AMT_CORR"),
        ("dev_vwap", "vol", "PRICEDEV_VOL_CORR"),
        ("dev_vwap", "amount_adj", "PRICEDEV_AMT_CORR"),
    )
    for x_column, y_column, name in correlation_pairs:
        aggregate = aggregate.merge(
            _correlation_by_code(detail, x_column, y_column, name), on="Code", how="left"
        )
    aggregate["slot"] = slot
    return aggregate


def build_m30_features(minutes: pd.DataFrame, daily_reference: pd.DataFrame) -> pd.DataFrame:
    """Build 70 formulas for each of eight signal-day half-hour bars."""

    _require_columns(
        minutes,
        ("Code", "Minute", "open", "high", "low", "close", "vol", "amount_adj"),
        name="normalized minute panel",
    )
    _require_columns(
        daily_reference,
        ("Code", "close_adj", "pre_close_adj", "volume_shares", "amount_adj"),
        name="daily reference",
    )
    minute_panel = minutes.copy()
    minute_panel["slot"] = minute_panel["Minute"].map(_MINUTE_TO_SLOT)
    minute_panel = minute_panel.dropna(subset=["slot"]).copy()
    minute_panel["slot"] = minute_panel["slot"].astype("int8")
    bars = [_aggregate_m30_slot(minute_panel, slot) for slot in range(1, 9)]
    bars = [bar for bar in bars if not bar.empty]
    if not bars:
        return pd.DataFrame(columns=("Code", *M30_FEATURE_COLUMNS))
    bar = pd.concat(bars, ignore_index=True).sort_values(["Code", "slot"], kind="stable")
    reference = daily_reference[
        ["Code", "close_adj", "pre_close_adj", "volume_shares", "amount_adj"]
    ].drop_duplicates("Code")
    bar = bar.merge(reference, on="Code", how="inner", validate="many_to_one")
    grouped = bar.groupby("Code", sort=False)
    bar["PREV_CLOSE_RAW"] = grouped["CLOSE_RAW"].shift(1)
    bar["PREV_VWAP_RAW"] = grouped["VWAP_RAW"].shift(1)
    bar["PREV_VOLUME_RAW"] = grouped["VOLUME_RAW"].shift(1)
    bar["PREV_AMOUNT_RAW"] = grouped["AMOUNT_RAW"].shift(1)
    first_slot = bar["slot"].eq(1)
    bar.loc[first_slot, "PREV_CLOSE_RAW"] = bar.loc[first_slot, "pre_close_adj"]
    bar["CUM_VOLUME_RAW"] = grouped["VOLUME_RAW"].cumsum()
    bar["CUM_AMOUNT_RAW"] = grouped["AMOUNT_RAW"].cumsum()
    bar["CUM_HIGH_RAW"] = grouped["HIGH_RAW"].cummax()
    bar["CUM_LOW_RAW"] = grouped["LOW_RAW"].cummin()
    bar["FIRST_OPEN_RAW"] = grouped["OPEN_RAW"].transform("first")
    bar["CUM_VWAP_RAW"] = safe_div(bar["CUM_AMOUNT_RAW"], bar["CUM_VOLUME_RAW"])

    high_low = bar["HIGH_RAW"] - bar["LOW_RAW"]
    body = bar["CLOSE_RAW"] - bar["OPEN_RAW"]
    feature = pd.DataFrame({"Code": bar["Code"].array, "slot": bar["slot"].array})
    feature["OPEN"] = safe_div(bar["OPEN_RAW"], bar["close_adj"])
    feature["CLOSE"] = safe_div(bar["CLOSE_RAW"], bar["close_adj"])
    feature["HIGH"] = safe_div(bar["HIGH_RAW"], bar["close_adj"])
    feature["LOW"] = safe_div(bar["LOW_RAW"], bar["close_adj"])
    feature["VWAP"] = safe_div(bar["VWAP_RAW"], bar["close_adj"])
    feature["VOLUME_FIRST"] = safe_div(bar["VOLUME_FIRST_RAW"], bar["volume_shares"])
    feature["VOLUME_LAST"] = safe_div(bar["VOLUME_LAST_RAW"], bar["volume_shares"])
    feature["VOLUME_MEAN"] = safe_div(bar["VOLUME_MEAN_RAW"], bar["volume_shares"])
    feature["AMOUNT_FIRST"] = safe_div(bar["AMOUNT_FIRST_RAW"], bar["amount_adj"])
    feature["AMOUNT_LAST"] = safe_div(bar["AMOUNT_LAST_RAW"], bar["amount_adj"])
    feature["AMOUNT_MEAN"] = safe_div(bar["AMOUNT_MEAN_RAW"], bar["amount_adj"])
    feature["VOL_SHARE"] = safe_div(bar["VOLUME_RAW"], bar["volume_shares"])
    feature["AMT_SHARE"] = safe_div(bar["AMOUNT_RAW"], bar["amount_adj"])
    feature["RET_OC"] = safe_div(bar["CLOSE_RAW"], bar["OPEN_RAW"]) - 1.0
    feature["RET_CC"] = safe_div(bar["CLOSE_RAW"], bar["PREV_CLOSE_RAW"]) - 1.0
    feature["GAP"] = safe_div(bar["OPEN_RAW"], bar["PREV_CLOSE_RAW"]) - 1.0
    feature["RET_VWAP"] = safe_div(bar["VWAP_RAW"], bar["PREV_VWAP_RAW"]) - 1.0
    feature["RANGE_OPEN"] = safe_div(high_low, bar["OPEN_RAW"])
    feature["BODY"] = safe_div(body, bar["OPEN_RAW"])
    feature["ABS_BODY"] = safe_div(np.abs(body), bar["OPEN_RAW"])
    feature["UPPER_SHADOW"] = safe_div(
        np.maximum(bar["HIGH_RAW"] - np.maximum(bar["OPEN_RAW"], bar["CLOSE_RAW"]), 0.0),
        bar["OPEN_RAW"],
    )
    feature["LOWER_SHADOW"] = safe_div(
        np.maximum(np.minimum(bar["OPEN_RAW"], bar["CLOSE_RAW"]) - bar["LOW_RAW"], 0.0),
        bar["OPEN_RAW"],
    )
    feature["BODY_RANGE_RATIO"] = safe_div(np.abs(body), high_low)
    feature["CLOSE_POS"] = safe_div(bar["CLOSE_RAW"] - bar["LOW_RAW"], high_low)
    feature["VWAP_POS"] = safe_div(bar["VWAP_RAW"] - bar["LOW_RAW"], high_low)
    feature["CLOSE_VWAP"] = safe_div(bar["CLOSE_RAW"], bar["VWAP_RAW"]) - 1.0
    feature["OPEN_VWAP"] = safe_div(bar["OPEN_RAW"], bar["VWAP_RAW"]) - 1.0
    feature["HIGH_VWAP"] = safe_div(bar["HIGH_RAW"], bar["VWAP_RAW"]) - 1.0
    feature["LOW_VWAP"] = safe_div(bar["LOW_RAW"], bar["VWAP_RAW"]) - 1.0
    feature["VWAP_OPEN"] = safe_div(bar["VWAP_RAW"], bar["OPEN_RAW"]) - 1.0
    feature["VOL_CHG"] = safe_div(bar["VOLUME_RAW"], bar["PREV_VOLUME_RAW"]) - 1.0
    feature["AMT_CHG"] = safe_div(bar["AMOUNT_RAW"], bar["PREV_AMOUNT_RAW"]) - 1.0
    feature["UP_VOL_RATIO"] = safe_div(bar["UP_VOL"], bar["VOLUME_RAW"])
    feature["DOWN_VOL_RATIO"] = safe_div(bar["DOWN_VOL"], bar["VOLUME_RAW"])
    feature["NET_UP_VOL_RATIO"] = safe_div(bar["UP_VOL"] - bar["DOWN_VOL"], bar["VOLUME_RAW"])
    feature["UP_AMT_RATIO"] = safe_div(bar["UP_AMT"], bar["AMOUNT_RAW"])
    feature["DOWN_AMT_RATIO"] = safe_div(bar["DOWN_AMT"], bar["AMOUNT_RAW"])
    feature["NET_UP_AMT_RATIO"] = safe_div(bar["UP_AMT"] - bar["DOWN_AMT"], bar["AMOUNT_RAW"])
    feature["RET_PER_AMT"] = safe_div(feature["RET_OC"], feature["AMT_SHARE"])
    feature["RET_PER_VOL"] = safe_div(feature["RET_OC"], feature["VOL_SHARE"])
    feature["ABS_RET_PER_AMT"] = safe_div(np.abs(feature["RET_OC"]), feature["AMT_SHARE"])
    feature["ABS_RET_PER_VOL"] = safe_div(np.abs(feature["RET_OC"]), feature["VOL_SHARE"])
    feature["RANGE_PER_AMT"] = safe_div(feature["RANGE_OPEN"], feature["AMT_SHARE"])
    feature["RANGE_PER_VOL"] = safe_div(feature["RANGE_OPEN"], feature["VOL_SHARE"])
    feature["MIN_RET_STD"] = bar["MIN_RET_STD"]
    feature["REALIZED_VOL"] = np.sqrt(bar["RET2_SUM"])
    feature["ABS_MIN_RET_MEAN"] = bar["ABS_MIN_RET_MEAN"]
    feature["MAX_MIN_RET"] = bar["MAX_MIN_RET"]
    feature["MIN_MIN_RET"] = bar["MIN_MIN_RET"]
    feature["UP_MIN_RATIO"] = safe_div(bar["UP_N"], bar["RET_N"])
    feature["DOWN_MIN_RATIO"] = safe_div(bar["DOWN_N"], bar["RET_N"])
    feature["FLAT_MIN_RATIO"] = safe_div(bar["FLAT_N"], bar["RET_N"])
    feature["CUM_RET"] = safe_div(bar["CLOSE_RAW"], bar["FIRST_OPEN_RAW"]) - 1.0
    feature["CUM_VWAP"] = safe_div(bar["CUM_VWAP_RAW"], bar["close_adj"])
    feature["CLOSE_CUM_VWAP"] = safe_div(bar["CLOSE_RAW"], bar["CUM_VWAP_RAW"]) - 1.0
    feature["CUM_CLOSE_POS"] = safe_div(
        bar["CLOSE_RAW"] - bar["CUM_LOW_RAW"], bar["CUM_HIGH_RAW"] - bar["CUM_LOW_RAW"]
    )
    feature["DIST_TO_CUM_HIGH"] = safe_div(bar["CLOSE_RAW"], bar["CUM_HIGH_RAW"]) - 1.0
    feature["DIST_TO_CUM_LOW"] = safe_div(bar["CLOSE_RAW"], bar["CUM_LOW_RAW"]) - 1.0
    feature["PRICE_DEV_VWAP_MEAN"] = bar["PRICE_DEV_VWAP_MEAN"]
    feature["PRICE_DEV_VWAP_ABS_MEAN"] = bar["PRICE_DEV_VWAP_ABS_MEAN"]
    feature["PRICE_DEV_VWAP_STD"] = bar["PRICE_DEV_VWAP_STD"]
    feature["PRICE_MAX_DEV_OPEN"] = bar["PRICE_MAX_DEV_OPEN"]
    feature["PRICE_MIN_DEV_OPEN"] = bar["PRICE_MIN_DEV_OPEN"]
    feature["PRICE_DEV_RANGE_OPEN"] = bar["PRICE_MAX_DEV_OPEN"] - bar["PRICE_MIN_DEV_OPEN"]
    for column in (
        "RET_VOL_CORR",
        "RET_AMT_CORR",
        "ABSRET_VOL_CORR",
        "ABSRET_AMT_CORR",
        "PRICEDEV_VOL_CORR",
        "PRICEDEV_AMT_CORR",
    ):
        feature[column] = bar[column]

    codes = sorted(feature["Code"].dropna().unique())
    data: dict[str, object] = {"Code": codes}
    for field in M30_FIELDS:
        wide = feature.pivot(index="Code", columns="slot", values=field).reindex(codes)
        for slot in range(1, 9):
            data[f"M30_{field}_{slot}"] = (
                wide[slot].to_numpy(dtype="float64") if slot in wide.columns else np.full(len(codes), np.nan)
            )
    output = pd.DataFrame(data)
    output[list(M30_FEATURE_COLUMNS)] = output[list(M30_FEATURE_COLUMNS)].astype("float32")
    return output[["Code", *M30_FEATURE_COLUMNS]]


def build_early_open_features(minutes: pd.DataFrame, daily_reference: pd.DataFrame) -> pd.DataFrame:
    """Build the 58 DE-only features from next-day 09:30 and 09:31 bars."""

    _require_columns(
        minutes,
        ("Code", "Minute", "open", "high", "low", "close", "vol", "amount_adj"),
        name="normalized early-open panel",
    )
    _require_columns(
        daily_reference,
        ("Code", "close_adj", "volume_shares", "amount_adj"),
        name="daily reference",
    )
    minute_panel = minutes[minutes["Minute"].isin((930, 931))].copy()
    if minute_panel.empty:
        return pd.DataFrame(columns=("Code", *EO_FEATURE_COLUMNS))
    minute_panel = minute_panel.sort_values(["Code", "Minute"], kind="stable")
    reference = daily_reference.rename(
        columns={"close_adj": "ref_close", "volume_shares": "ref_volume", "amount_adj": "ref_amount"}
    )[["Code", "ref_close", "ref_volume", "ref_amount"]].drop_duplicates("Code")
    grouped = minute_panel.groupby("Code", sort=False)
    aggregate = grouped.agg(
        EO_OPEN_RAW=("open", "first"),
        EO_CLOSE_RAW=("close", "last"),
        EO_HIGH_RAW=("high", "max"),
        EO_LOW_RAW=("low", "min"),
        EO_VOLUME_RAW=("vol", "sum"),
        EO_AMOUNT_RAW=("amount_adj", "sum"),
        EO_VOLUME_FIRST_RAW=("vol", "first"),
        EO_VOLUME_LAST_RAW=("vol", "last"),
        EO_VOLUME_MEAN_RAW=("vol", "mean"),
        EO_AMOUNT_FIRST_RAW=("amount_adj", "first"),
        EO_AMOUNT_LAST_RAW=("amount_adj", "last"),
        EO_AMOUNT_MEAN_RAW=("amount_adj", "mean"),
    ).reset_index()
    aggregate["EO_VWAP_RAW"] = safe_div(aggregate["EO_AMOUNT_RAW"], aggregate["EO_VOLUME_RAW"])
    auction = minute_panel[minute_panel["Minute"].eq(930)][["Code", "close", "vol", "amount_adj"]].rename(
        columns={"close": "AUCTION_PRICE", "vol": "AUCTION_VOL", "amount_adj": "AUCTION_AMT"}
    )
    auction = auction.drop_duplicates("Code", keep="last")
    aggregate = aggregate.merge(auction, on="Code", how="left", validate="one_to_one")
    aggregate = aggregate.merge(reference, on="Code", how="left", validate="one_to_one")
    detail = minute_panel.merge(
        aggregate[["Code", "EO_VWAP_RAW", "EO_OPEN_RAW"]], on="Code", how="left", validate="many_to_one"
    )
    detail["ret"] = detail.groupby("Code", sort=False)["close"].pct_change(fill_method=None)
    detail["absret"] = detail["ret"].abs()
    detail["ret2"] = detail["ret"] ** 2
    detail["dev_vwap"] = safe_div(detail["close"], detail["EO_VWAP_RAW"]) - 1.0
    detail["dev_open"] = safe_div(detail["close"], detail["EO_OPEN_RAW"]) - 1.0
    detail["up_vol"] = np.where(detail["ret"] > 0, detail["vol"], 0.0)
    detail["down_vol"] = np.where(detail["ret"] < 0, detail["vol"], 0.0)
    detail["up_amt"] = np.where(detail["ret"] > 0, detail["amount_adj"], 0.0)
    detail["down_amt"] = np.where(detail["ret"] < 0, detail["amount_adj"], 0.0)
    stats = (
        detail.groupby("Code", sort=False)
        .agg(
            RET_N=("ret", "count"),
            RET2_SUM=("ret2", "sum"),
            ABS_RET_MEAN=("absret", "mean"),
            MAX_RET=("ret", "max"),
            MIN_RET=("ret", "min"),
            UP_N=("ret", lambda values: float((values > 0).sum())),
            DOWN_N=("ret", lambda values: float((values < 0).sum())),
            FLAT_N=("ret", lambda values: float((values == 0).sum())),
            UP_VOL=("up_vol", "sum"),
            DOWN_VOL=("down_vol", "sum"),
            UP_AMT=("up_amt", "sum"),
            DOWN_AMT=("down_amt", "sum"),
            DEV_MEAN=("dev_vwap", "mean"),
            DEV_ABS_MEAN=("dev_vwap", lambda values: values.abs().mean()),
            DEV_STD=("dev_vwap", "std"),
            DEV_MAX_OPEN=("dev_open", "max"),
            DEV_MIN_OPEN=("dev_open", "min"),
        )
        .reset_index()
    )
    aggregate = aggregate.merge(stats, on="Code", how="left", validate="one_to_one")
    for x_column, y_column, name in (
        ("ret", "vol", "RET_VOL_CORR"),
        ("ret", "amount_adj", "RET_AMT_CORR"),
        ("absret", "vol", "ABSRET_VOL_CORR"),
        ("absret", "amount_adj", "ABSRET_AMT_CORR"),
        ("dev_vwap", "vol", "PRICEDEV_VOL_CORR"),
        ("dev_vwap", "amount_adj", "PRICEDEV_AMT_CORR"),
    ):
        aggregate = aggregate.merge(
            _correlation_by_code(detail, x_column, y_column, name), on="Code", how="left"
        )

    high_low = aggregate["EO_HIGH_RAW"] - aggregate["EO_LOW_RAW"]
    body = aggregate["EO_CLOSE_RAW"] - aggregate["EO_OPEN_RAW"]
    output = pd.DataFrame({"Code": aggregate["Code"].array})
    output["EO_OPEN"] = safe_div(aggregate["EO_OPEN_RAW"], aggregate["ref_close"])
    output["EO_CLOSE"] = safe_div(aggregate["EO_CLOSE_RAW"], aggregate["ref_close"])
    output["EO_HIGH"] = safe_div(aggregate["EO_HIGH_RAW"], aggregate["ref_close"])
    output["EO_LOW"] = safe_div(aggregate["EO_LOW_RAW"], aggregate["ref_close"])
    output["EO_VWAP"] = safe_div(aggregate["EO_VWAP_RAW"], aggregate["ref_close"])
    output["EO_RET_PREV_CLOSE"] = safe_div(aggregate["EO_CLOSE_RAW"], aggregate["ref_close"]) - 1.0
    output["EO_VWAP_PREV_CLOSE"] = safe_div(aggregate["EO_VWAP_RAW"], aggregate["ref_close"]) - 1.0
    output["EO_AUCTION_RET_PREV_CLOSE"] = safe_div(aggregate["AUCTION_PRICE"], aggregate["ref_close"]) - 1.0
    output["EO_VOLUME_FIRST"] = safe_div(aggregate["EO_VOLUME_FIRST_RAW"], aggregate["ref_volume"])
    output["EO_VOLUME_LAST"] = safe_div(aggregate["EO_VOLUME_LAST_RAW"], aggregate["ref_volume"])
    output["EO_VOLUME_MEAN"] = safe_div(aggregate["EO_VOLUME_MEAN_RAW"], aggregate["ref_volume"])
    output["EO_AMOUNT_FIRST"] = safe_div(aggregate["EO_AMOUNT_FIRST_RAW"], aggregate["ref_amount"])
    output["EO_AMOUNT_LAST"] = safe_div(aggregate["EO_AMOUNT_LAST_RAW"], aggregate["ref_amount"])
    output["EO_AMOUNT_MEAN"] = safe_div(aggregate["EO_AMOUNT_MEAN_RAW"], aggregate["ref_amount"])
    output["EO_VOL_SHARE_PREV_DAY"] = safe_div(aggregate["EO_VOLUME_RAW"], aggregate["ref_volume"])
    output["EO_AMT_SHARE_PREV_DAY"] = safe_div(aggregate["EO_AMOUNT_RAW"], aggregate["ref_amount"])
    output["EO_RET_OC"] = safe_div(aggregate["EO_CLOSE_RAW"], aggregate["EO_OPEN_RAW"]) - 1.0
    output["EO_RANGE_OPEN"] = safe_div(high_low, aggregate["EO_OPEN_RAW"])
    output["EO_BODY"] = safe_div(body, aggregate["EO_OPEN_RAW"])
    output["EO_ABS_BODY"] = safe_div(np.abs(body), aggregate["EO_OPEN_RAW"])
    output["EO_UPPER_SHADOW"] = safe_div(
        np.maximum(
            aggregate["EO_HIGH_RAW"] - np.maximum(aggregate["EO_OPEN_RAW"], aggregate["EO_CLOSE_RAW"]),
            0.0,
        ),
        aggregate["EO_OPEN_RAW"],
    )
    output["EO_LOWER_SHADOW"] = safe_div(
        np.maximum(
            np.minimum(aggregate["EO_OPEN_RAW"], aggregate["EO_CLOSE_RAW"]) - aggregate["EO_LOW_RAW"],
            0.0,
        ),
        aggregate["EO_OPEN_RAW"],
    )
    output["EO_BODY_RANGE_RATIO"] = safe_div(np.abs(body), high_low)
    output["EO_CLOSE_POS"] = safe_div(aggregate["EO_CLOSE_RAW"] - aggregate["EO_LOW_RAW"], high_low)
    output["EO_VWAP_POS"] = safe_div(aggregate["EO_VWAP_RAW"] - aggregate["EO_LOW_RAW"], high_low)
    output["EO_CLOSE_VWAP"] = safe_div(aggregate["EO_CLOSE_RAW"], aggregate["EO_VWAP_RAW"]) - 1.0
    output["EO_OPEN_VWAP"] = safe_div(aggregate["EO_OPEN_RAW"], aggregate["EO_VWAP_RAW"]) - 1.0
    output["EO_HIGH_VWAP"] = safe_div(aggregate["EO_HIGH_RAW"], aggregate["EO_VWAP_RAW"]) - 1.0
    output["EO_LOW_VWAP"] = safe_div(aggregate["EO_LOW_RAW"], aggregate["EO_VWAP_RAW"]) - 1.0
    output["EO_VWAP_OPEN"] = safe_div(aggregate["EO_VWAP_RAW"], aggregate["EO_OPEN_RAW"]) - 1.0
    output["EO_UP_VOL_RATIO"] = safe_div(aggregate["UP_VOL"], aggregate["EO_VOLUME_RAW"])
    output["EO_DOWN_VOL_RATIO"] = safe_div(aggregate["DOWN_VOL"], aggregate["EO_VOLUME_RAW"])
    output["EO_NET_UP_VOL_RATIO"] = safe_div(
        aggregate["UP_VOL"] - aggregate["DOWN_VOL"], aggregate["EO_VOLUME_RAW"]
    )
    output["EO_UP_AMT_RATIO"] = safe_div(aggregate["UP_AMT"], aggregate["EO_AMOUNT_RAW"])
    output["EO_DOWN_AMT_RATIO"] = safe_div(aggregate["DOWN_AMT"], aggregate["EO_AMOUNT_RAW"])
    output["EO_NET_UP_AMT_RATIO"] = safe_div(
        aggregate["UP_AMT"] - aggregate["DOWN_AMT"], aggregate["EO_AMOUNT_RAW"]
    )
    output["EO_REALIZED_VOL"] = np.sqrt(aggregate["RET2_SUM"])
    output["EO_ABS_MIN_RET_MEAN"] = aggregate["ABS_RET_MEAN"]
    output["EO_MAX_MIN_RET"] = aggregate["MAX_RET"]
    output["EO_MIN_MIN_RET"] = aggregate["MIN_RET"]
    output["EO_UP_MIN_RATIO"] = safe_div(aggregate["UP_N"], aggregate["RET_N"])
    output["EO_DOWN_MIN_RATIO"] = safe_div(aggregate["DOWN_N"], aggregate["RET_N"])
    output["EO_FLAT_MIN_RATIO"] = safe_div(aggregate["FLAT_N"], aggregate["RET_N"])
    output["EO_PRICE_DEV_VWAP_MEAN"] = aggregate["DEV_MEAN"]
    output["EO_PRICE_DEV_VWAP_ABS_MEAN"] = aggregate["DEV_ABS_MEAN"]
    output["EO_PRICE_DEV_VWAP_STD"] = aggregate["DEV_STD"]
    output["EO_PRICE_MAX_DEV_OPEN"] = aggregate["DEV_MAX_OPEN"]
    output["EO_PRICE_MIN_DEV_OPEN"] = aggregate["DEV_MIN_OPEN"]
    output["EO_PRICE_DEV_RANGE_OPEN"] = aggregate["DEV_MAX_OPEN"] - aggregate["DEV_MIN_OPEN"]
    output["EO_RET_VOL_CORR"] = aggregate["RET_VOL_CORR"]
    output["EO_RET_AMT_CORR"] = aggregate["RET_AMT_CORR"]
    output["EO_ABSRET_VOL_CORR"] = aggregate["ABSRET_VOL_CORR"]
    output["EO_ABSRET_AMT_CORR"] = aggregate["ABSRET_AMT_CORR"]
    output["EO_PRICEDEV_VOL_CORR"] = aggregate["PRICEDEV_VOL_CORR"]
    output["EO_PRICEDEV_AMT_CORR"] = aggregate["PRICEDEV_AMT_CORR"]
    output["EO_AUCTION_VOL_SHARE"] = safe_div(aggregate["AUCTION_VOL"], aggregate["EO_VOLUME_RAW"])
    output["EO_AUCTION_AMT_SHARE"] = safe_div(aggregate["AUCTION_AMT"], aggregate["EO_AMOUNT_RAW"])
    output["EO_AUCTION_PRICE_DEV_VWAP"] = safe_div(aggregate["AUCTION_PRICE"], aggregate["EO_VWAP_RAW"]) - 1.0
    output[list(EO_FEATURE_COLUMNS)] = output[list(EO_FEATURE_COLUMNS)].astype("float32")
    return output[["Code", *EO_FEATURE_COLUMNS]]


def build_raw_feature_panel(
    raw_daily: pd.DataFrame,
    signal_minutes: pd.DataFrame,
    *,
    signal_date: object,
    protocol: str = "o2o",
    early_open_minutes: pd.DataFrame | None = None,
    code_prefixes: tuple[str, ...] = DEFAULT_CODE_PREFIXES,
) -> pd.DataFrame:
    """Build one signal-date O2O (599) or DE (657) raw feature panel."""

    protocol_key = protocol.strip().lower()
    if protocol_key not in {"o2o", "de"}:
        raise ValueError("protocol must be 'o2o' or 'de'")
    normalized_date = normalize_date_series(pd.Series([signal_date])).iloc[0]
    if pd.isna(normalized_date):
        raise ValueError(f"invalid signal date: {signal_date!r}")
    date = int(normalized_date)

    daily = prepare_daily_panel(raw_daily, code_prefixes=code_prefixes)
    daily_reference = daily[daily["Date"].eq(date)].copy()
    if daily_reference.empty:
        raise ValueError(f"daily panel does not contain signal date {date}")
    daily_features = build_daily_features(daily)
    daily_features = daily_features[daily_features["Date"].eq(date)].drop(columns="Date")
    minutes = prepare_minute_panel(signal_minutes, code_prefixes=code_prefixes)
    m30_features = build_m30_features(minutes, daily_reference)
    if m30_features.empty:
        raise ValueError("signal-day panel contains no M30 observations")
    output = m30_features.merge(daily_features, on="Code", how="inner", validate="one_to_one")

    expected_columns = O2O_RAW_FEATURE_COLUMNS
    if protocol_key == "de":
        if early_open_minutes is None:
            raise ValueError("DE requires next-session 09:30/09:31 bars")
        early_minutes = prepare_minute_panel(early_open_minutes, code_prefixes=code_prefixes)
        early_features = build_early_open_features(early_minutes, daily_reference)
        if early_features.empty:
            raise ValueError("DE early-open panel contains neither 09:30 nor 09:31 observations")
        output = output.merge(early_features, on="Code", how="left", validate="one_to_one")
        expected_columns = DE_RAW_FEATURE_COLUMNS

    output.insert(1, "Date", date)
    output = output[["Code", "Date", *expected_columns]].copy()
    output[list(expected_columns)] = (
        output[list(expected_columns)].replace([np.inf, -np.inf], np.nan).astype("float32")
    )
    return output.sort_values("Code", kind="stable").reset_index(drop=True)


def build_o2o_raw_features(
    raw_daily: pd.DataFrame,
    signal_minutes: pd.DataFrame,
    *,
    signal_date: object,
    code_prefixes: tuple[str, ...] = DEFAULT_CODE_PREFIXES,
) -> pd.DataFrame:
    """Convenience wrapper returning exactly 599 raw O2O features."""

    return build_raw_feature_panel(
        raw_daily,
        signal_minutes,
        signal_date=signal_date,
        protocol="o2o",
        code_prefixes=code_prefixes,
    )


def build_de_raw_features(
    raw_daily: pd.DataFrame,
    signal_minutes: pd.DataFrame,
    early_open_minutes: pd.DataFrame,
    *,
    signal_date: object,
    code_prefixes: tuple[str, ...] = DEFAULT_CODE_PREFIXES,
) -> pd.DataFrame:
    """Convenience wrapper returning exactly 657 raw DE features."""

    return build_raw_feature_panel(
        raw_daily,
        signal_minutes,
        signal_date=signal_date,
        protocol="de",
        early_open_minutes=early_open_minutes,
        code_prefixes=code_prefixes,
    )
