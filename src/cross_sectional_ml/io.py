"""Small, strict I/O helpers for dated cross-sectional panels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

KEY_COLUMNS = ("Code", "Date")


def normalize_code_series(values: pd.Series) -> pd.Series:
    """Return exchange-free security identifiers.

    Numeric A-share codes are padded to six digits.  Short alphanumeric
    identifiers are retained so the same panel contract also works with
    synthetic and external universes.
    """

    text = values.astype("string").str.strip().str.split(".", n=1).str[0]
    numeric = text.str.fullmatch(r"\d{1,6}")
    symbolic = text.str.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
    normalized = text.where(symbolic).str.upper()
    normalized = normalized.where(~numeric, text.str.zfill(6))
    return normalized.where(numeric | symbolic)


def normalize_date_series(values: pd.Series) -> pd.Series:
    """Normalize dates to nullable ``YYYYMMDD`` integers."""

    text = values.astype("string").str.strip().str.replace(r"\D", "", regex=True).str[:8]
    valid = text.str.fullmatch(r"\d{8}")
    return pd.to_numeric(text.where(valid), errors="coerce").astype("Int64")


def normalize_datetime_series(values: pd.Series) -> pd.Series:
    """Normalize common date representations to midnight timestamps."""

    dates = normalize_date_series(values)
    return pd.to_datetime(dates.astype("string"), format="%Y%m%d", errors="coerce")


def normalize_minute_series(values: pd.Series) -> pd.Series:
    """Normalize HHMM integers and standard clock strings to nullable HHMM."""

    text = values.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)
    compact = text.where(text.str.fullmatch(r"\d{1,4}"))
    result = pd.to_numeric(compact, errors="coerce").astype("Int64")
    clock = text.str.extract(r"^(\d{1,2}):(\d{2})(?::\d{2}(?:\.\d+)?)?$")
    hour = pd.to_numeric(clock[0], errors="coerce")
    minute = pd.to_numeric(clock[1], errors="coerce")
    parsed_clock = (hour * 100 + minute).where(hour.between(0, 23) & minute.between(0, 59))
    result = result.fillna(parsed_clock).astype("Int64")
    hour_hhmm = result // 100
    minute_hhmm = result % 100
    return result.where(hour_hhmm.between(0, 23) & minute_hhmm.between(0, 59))


def normalize_panel_keys(frame: pd.DataFrame, *, require_date: bool = True) -> pd.DataFrame:
    """Copy *frame*, normalize its key columns, and enforce key uniqueness."""

    required = {"Code"}
    if require_date:
        required.add("Date")
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing panel key columns: {missing}")

    out = frame.copy()
    out["Code"] = normalize_code_series(out["Code"])
    if require_date:
        out["Date"] = normalize_date_series(out["Date"])
    if out[list(required)].isna().any(axis=None):
        raise ValueError("panel contains null or invalid keys")
    key_columns = ["Code", "Date"] if require_date else ["Code"]
    if out.duplicated(key_columns).any():
        raise ValueError(f"panel contains duplicate keys: {key_columns}")
    if require_date:
        out["Date"] = out["Date"].astype("int64")
    return out


def read_parquet(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    normalize_keys: bool = False,
) -> pd.DataFrame:
    """Read one Parquet panel, optionally with an exact column projection."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    requested = list(columns) if columns is not None else None
    frame = pd.read_parquet(source, columns=requested)
    if requested is not None and list(frame.columns) != requested:
        raise ValueError(
            f"Parquet projection order differs from request: expected {requested}, got {list(frame.columns)}"
        )
    if normalize_keys:
        frame = normalize_panel_keys(frame, require_date="Date" in frame.columns)
    return frame


def write_parquet(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write a DataFrame as Parquet and return the resolved output path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=index)
    return target


def list_dated_parquet_files(directory: str | Path) -> dict[int, Path]:
    """Return sorted ``YYYYMMDD.parquet`` files keyed by their date."""

    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(root)
    dated: dict[int, Path] = {}
    for path in sorted(root.glob("*.parquet")):
        if len(path.stem) == 8 and path.stem.isdigit():
            date = int(path.stem)
            if date in dated:
                raise ValueError(f"duplicate dated Parquet file: {date}")
            dated[date] = path
    return dated


def read_many_parquet(
    paths: Iterable[str | Path],
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read and concatenate a finite collection of identically shaped panels."""

    parts = [read_parquet(path, columns=columns) for path in paths]
    if not parts:
        return pd.DataFrame(columns=list(columns or ()))
    return pd.concat(parts, ignore_index=True)
