from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_frame(path: str | Path, *, code_column: str = "Code") -> pd.DataFrame:
    source = Path(path)
    if source.is_dir():
        parts = [pd.read_parquet(file) for file in sorted(source.glob("*.parquet"))]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source, dtype={code_column: "string"})
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(source, sep="\t", dtype={code_column: "string"})
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(source)
    raise ValueError(f"unsupported table: {source}")


def read_dated_directory(path: str | Path, date_column: str = "Date") -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for file in sorted(Path(path).glob("*.parquet")):
        frame = pd.read_parquet(file)
        if date_column not in frame.columns and date_column.lower() not in frame.columns:
            frame[date_column] = file.stem
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def read_minute_session(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.is_file():
        return read_frame(source)
    parts: list[pd.DataFrame] = []
    for file in sorted(source.glob("*.parquet")):
        frame = pd.read_parquet(file)
        if "Minute" not in frame.columns and "minute" not in frame.columns:
            frame["Minute"] = file.stem
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def write_json(value: object, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    return target


def read_active_split_dates(path: str | Path, fold: str) -> dict[str, list[int]]:
    roles = read_frame(path)
    purged = roles["purged"].astype("string").str.lower().isin({"true", "1"})
    active = roles[roles["fold_id"].eq(fold) & ~purged]
    return {
        split: active.loc[active["split"].eq(split), "signal_date"].astype(int).tolist()
        for split in ("train", "validation", "test")
    }
