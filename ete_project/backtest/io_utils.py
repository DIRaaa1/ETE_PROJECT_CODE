import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TABLE_SUFFIXES = {".csv", ".gz", ".parquet", ".pkl", ".pickle", ".feather"}


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_path(value: Optional[str], base_dir: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    p = Path(text)
    if p.is_absolute():
        return str(p)
    return str((Path(base_dir) / p).resolve())


def sanitize_name(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "signal"


def normalize_date_value(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    raw = re.sub(r"\D", "", str(value))
    if len(raw) >= 8:
        return raw[:8]
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt).strftime("%Y%m%d")


def normalize_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = out[date_col].map(normalize_date_value)
    out = out[out[date_col].notna()].copy()
    return out


def read_pickle_compat(path: str) -> Any:
    class NPCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core")
            return super().find_class(module, name)

    with open(path, "rb") as f:
        return NPCompatUnpickler(f).load()


def read_table(path: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".gz":
        suffix = "".join(Path(path).suffixes[-2:]).lower()
    if suffix in {".pkl", ".pickle"}:
        obj = read_pickle_compat(path)
        if not isinstance(obj, pd.DataFrame):
            obj = pd.DataFrame(obj)
        if columns is not None:
            keep = [c for c in columns if c in obj.columns]
            obj = obj[keep]
        return obj
    if suffix in {".csv", ".csv.gz"}:
        return pd.read_csv(path, usecols=columns, encoding="utf-8-sig")
    if suffix == ".feather":
        return pd.read_feather(path, columns=columns)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            try:
                import polars as pl
            except Exception as exc:
                raise RuntimeError(f"Cannot read parquet file {path}") from exc
            frame = pl.read_parquet(path, columns=list(columns) if columns is not None else None)
            return pd.DataFrame(frame.to_dicts())
    raise ValueError(f"Unsupported table format: {path}")


def table_files_under(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if not p.is_dir():
        return []
    files = []
    for item in p.rglob("*"):
        if item.is_file() and item.suffix.lower() in TABLE_SUFFIXES:
            files.append(str(item))
    return sorted(files)


def list_prediction_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if not p.is_dir():
        return []
    files = sorted(str(x) for x in p.rglob("predictions.csv") if x.is_file())
    if files:
        return files
    return sorted(str(x) for x in p.rglob("*.csv") if x.is_file())


def infer_signal_name(path: str, used_names: Iterable[str]) -> str:
    p = Path(path)
    parts = list(p.parts)
    name = p.stem
    if p.name.lower() == "predictions.csv":
        parent = p.parent.name
        grand = p.parent.parent.name if p.parent.parent != p.parent else ""
        name = grand if parent.startswith("results_") and grand else parent
    base = sanitize_name(name)
    used = set(used_names)
    if base not in used:
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


def load_predictions(
    prediction_path: str,
    date_col: str,
    symbol_col: str,
    signal_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    files = list_prediction_files(prediction_path)
    if not files:
        raise FileNotFoundError(f"No prediction files found under {prediction_path}")

    merged = None
    loaded_signals: List[str] = []
    loaded_files: List[str] = []
    multi_file = len(files) > 1

    for file_path in files:
        df = read_table(file_path)
        missing = [c for c in [date_col, symbol_col] if c not in df.columns]
        if missing:
            raise KeyError(f"{file_path} is missing required columns: {missing}")

        if signal_cols:
            candidates = [c for c in signal_cols if c in df.columns]
        elif "y_pred" in df.columns:
            candidates = ["y_pred"]
        else:
            candidates = [
                c for c in df.columns
                if c not in {date_col, symbol_col} and pd.api.types.is_numeric_dtype(df[c])
            ]

        if not candidates:
            raise KeyError(f"{file_path} does not contain usable signal columns")

        keep = df[[date_col, symbol_col] + candidates].copy()
        keep = normalize_date_column(keep, date_col)
        keep[symbol_col] = keep[symbol_col].astype(str)

        rename_map = {}
        if multi_file and candidates == ["y_pred"]:
            signal_name = infer_signal_name(file_path, loaded_signals)
            rename_map["y_pred"] = signal_name
        else:
            for col in candidates:
                signal_name = sanitize_name(col)
                if signal_name in loaded_signals:
                    signal_name = infer_signal_name(f"{Path(file_path).stem}_{col}", loaded_signals)
                rename_map[col] = signal_name

        keep = keep.rename(columns=rename_map)
        current_signals = [rename_map[c] for c in candidates]
        loaded_signals.extend(current_signals)
        loaded_files.append(file_path)

        if merged is None:
            merged = keep
        else:
            merged = merged.merge(keep, on=[date_col, symbol_col], how="outer")

    assert merged is not None
    return merged, loaded_signals, loaded_files


def daily_file_candidates(directory: str, date_value: str, file_pattern: Optional[str]) -> List[str]:
    root = Path(directory)
    candidates = []
    if file_pattern:
        candidates.append(str(root / file_pattern.format(date=date_value)))
    for suffix in [".parquet", ".csv", ".csv.gz", ".pkl", ".pickle", ".feather"]:
        candidates.append(str(root / f"{date_value}{suffix}"))
        candidates.append(str(root / f"*{date_value}*{suffix}"))
    out = []
    for item in candidates:
        if "*" in item:
            out.extend(str(x) for x in root.glob(Path(item).name) if x.is_file())
        elif Path(item).is_file():
            out.append(item)
    seen = set()
    unique = []
    for item in out:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def load_external_frame(
    path: Optional[str],
    directory: Optional[str],
    date_col: str,
    symbol_col: str,
    needed_cols: Sequence[str],
    prediction_dates: Sequence[str],
    file_pattern: Optional[str] = None,
) -> pd.DataFrame:
    needed = list(dict.fromkeys([date_col, symbol_col] + list(needed_cols)))
    frames: List[pd.DataFrame] = []

    if path:
        source = Path(path)
        if source.is_file():
            frames.append(read_table(str(source)))
        elif source.is_dir():
            for file_path in table_files_under(str(source)):
                frames.append(read_table(file_path))

    if directory:
        for date_value in sorted(set(prediction_dates)):
            candidates = daily_file_candidates(directory, date_value, file_pattern)
            if not candidates:
                continue
            df = read_table(candidates[0])
            if date_col not in df.columns:
                df[date_col] = date_value
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=needed)

    trimmed = []
    for df in frames:
        if date_col not in df.columns:
            continue
        if symbol_col not in df.columns:
            continue
        cols = [c for c in needed if c in df.columns]
        if len(cols) < 2:
            continue
        cur = normalize_date_column(df[cols].copy(), date_col)
        cur[symbol_col] = cur[symbol_col].astype(str)
        trimmed.append(cur)

    if not trimmed:
        return pd.DataFrame(columns=needed)

    out = pd.concat(trimmed, ignore_index=True, copy=False)
    out = out.drop_duplicates([date_col, symbol_col], keep="last")
    return out


def infer_target_cols(df: pd.DataFrame, excluded_cols: Sequence[str]) -> List[str]:
    excluded = set(excluded_cols)
    hints = ["target", "label", "return", "ret", "fret", "oto", "pnl"]
    cols = []
    for col in df.columns:
        if col in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        low = str(col).lower()
        if any(h in low for h in hints):
            cols.append(col)
    return cols


def apply_date_filter(df: pd.DataFrame, date_col: str, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if start_date:
        s = normalize_date_value(start_date)
        out = out[out[date_col] >= s]
    if end_date:
        e = normalize_date_value(end_date)
        out = out[out[date_col] <= e]
    return out.copy()


def collect_backtest_frame(config: Dict[str, Any], base_dir: str) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
    date_col = str(config.get("date_col", "Date"))
    symbol_col = str(config.get("symbol_col", "Code"))
    prediction_path = resolve_path(config.get("prediction_path"), base_dir)
    if prediction_path is None:
        raise ValueError("prediction_path is required")

    signal_override = config.get("signal_cols")
    signals_arg = list(signal_override) if signal_override else None
    df, signal_cols, prediction_files = load_predictions(prediction_path, date_col, symbol_col, signals_arg)
    dates = sorted(df[date_col].dropna().astype(str).unique().tolist())

    target_cols = list(config.get("target_cols") or [])
    bet_cols = list(config.get("bet_size_cols") or [])
    external_needed = list(dict.fromkeys(target_cols + bet_cols))

    external = load_external_frame(
        path=resolve_path(config.get("target_path"), base_dir),
        directory=resolve_path(config.get("target_dir") or config.get("label_dir"), base_dir),
        date_col=date_col,
        symbol_col=symbol_col,
        needed_cols=external_needed,
        prediction_dates=dates,
        file_pattern=config.get("target_file_pattern") or config.get("label_file_pattern"),
    )
    if not external.empty:
        external_cols = [c for c in external.columns if c not in {date_col, symbol_col}]
        overlap = [c for c in external_cols if c in df.columns]
        if overlap:
            df = df.drop(columns=overlap)
        df = df.merge(external, on=[date_col, symbol_col], how="left")

    bet_external = load_external_frame(
        path=resolve_path(config.get("bet_size_path"), base_dir),
        directory=resolve_path(config.get("bet_size_dir"), base_dir),
        date_col=date_col,
        symbol_col=symbol_col,
        needed_cols=bet_cols,
        prediction_dates=dates,
        file_pattern=config.get("bet_size_file_pattern"),
    )
    if not bet_external.empty:
        external_cols = [c for c in bet_external.columns if c not in {date_col, symbol_col}]
        overlap = [c for c in external_cols if c in df.columns]
        if overlap:
            df = df.drop(columns=overlap)
        df = df.merge(bet_external, on=[date_col, symbol_col], how="left")

    if not target_cols:
        target_cols = infer_target_cols(df, [date_col, symbol_col] + signal_cols + bet_cols)
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target columns after merge: {missing_targets}")

    if not bet_cols:
        unit_col = str(config.get("unit_bet_col", "unit_bet"))
        df[unit_col] = float(config.get("default_bet_size", 1.0))
        bet_cols = [unit_col]
    else:
        missing_bets = [c for c in bet_cols if c not in df.columns]
        if missing_bets:
            raise KeyError(f"Missing bet size columns after merge: {missing_bets}")

    df = apply_date_filter(df, date_col, config.get("start_date"), config.get("end_date"))
    numeric_cols = signal_cols + target_cols + bet_cols
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, signal_cols, target_cols, bet_cols, prediction_files
