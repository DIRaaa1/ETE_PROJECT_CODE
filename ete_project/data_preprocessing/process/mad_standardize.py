import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


MAD_NORMAL_SCALE = 1.4826


def list_daily_parquet(input_dir: Path, start_date: Optional[int], end_date: Optional[int]) -> List[Path]:
    paths = []
    for path in sorted(input_dir.glob("*.parquet")):
        if not path.stem.isdigit():
            continue
        day = int(path.stem)
        if start_date is not None and day < int(start_date):
            continue
        if end_date is not None and day > int(end_date):
            continue
        paths.append(path)
    return paths


def parse_columns(value: Optional[str]) -> Optional[List[str]]:
    if value is None or str(value).strip() == "":
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def select_numeric_columns(df: pd.DataFrame, exclude_cols: Iterable[str], feature_prefix: str = "") -> List[str]:
    exclude = set(str(c) for c in exclude_cols if c is not None and str(c) != "")
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if feature_prefix and not str(col).startswith(feature_prefix):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(str(col))
    return cols


def mad_zscore_block(values: np.ndarray, mad_multiplier: float = 3.0, fill_value: float = 0.0) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must be a 2D array")
    if values.shape[0] == 0 or values.shape[1] == 0:
        return values.astype(np.float32, copy=True)

    x = values.astype(np.float64, copy=True)
    finite = np.isfinite(x)
    with np.errstate(all="ignore"):
        median = np.nanmedian(np.where(finite, x, np.nan), axis=0)
        mad = np.nanmedian(np.abs(np.where(finite, x, np.nan) - median), axis=0)

    scale = MAD_NORMAL_SCALE * mad
    width = float(mad_multiplier) * scale
    lower = median - width
    upper = median + width
    clipped = np.clip(x, lower, upper)

    out = np.full_like(clipped, float(fill_value), dtype=np.float64)
    valid = np.isfinite(scale) & (scale > 0)
    if np.any(valid):
        out[:, valid] = (clipped[:, valid] - median[valid]) / scale[valid]
    out[~np.isfinite(out)] = float(fill_value)
    return out.astype(np.float32)


def mad_standardize_frame(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: Sequence[str],
    mad_multiplier: float,
    output_suffix: str,
) -> pd.DataFrame:
    if group_col not in df.columns:
        raise KeyError(f"group column not found: {group_col}")
    feature_cols = [str(c) for c in feature_cols]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns not found: {missing[:10]}")

    result = df.copy()
    output_cols = [f"{col}{output_suffix}" for col in feature_cols]
    values = result.loc[:, feature_cols].to_numpy(dtype=np.float64, copy=True)
    for col in output_cols:
        result[col] = np.float32(0.0)

    standardized = np.empty((len(result), len(feature_cols)), dtype=np.float32)
    groups = result.groupby(group_col, sort=False).indices
    for indexer in groups.values():
        standardized[indexer, :] = mad_zscore_block(values[indexer, :], mad_multiplier=mad_multiplier)

    result.loc[:, output_cols] = standardized
    return result


def cross_section_standardize_dir(
    input_dir: Path,
    output_dir: Path,
    feature_cols: Optional[Sequence[str]],
    date_col: str,
    code_col: str,
    label_cols: Sequence[str],
    feature_prefix: str,
    mad_multiplier: float,
    output_suffix: str,
    start_date: Optional[int],
    end_date: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = list_daily_parquet(input_dir, start_date, end_date)
    if not paths:
        raise FileNotFoundError(f"no daily parquet files found in {input_dir}")

    selected_cols = list(feature_cols) if feature_cols is not None else None
    for path in paths:
        df = pd.read_parquet(path)
        if selected_cols is None:
            selected = select_numeric_columns(df, [date_col, code_col, *label_cols], feature_prefix=feature_prefix)
        else:
            selected = selected_cols
        out = mad_standardize_frame(
            df=df,
            group_col=date_col,
            feature_cols=selected,
            mad_multiplier=mad_multiplier,
            output_suffix=output_suffix,
        )
        out.to_parquet(output_dir / path.name, index=False)


def time_series_standardize_dir(
    input_dir: Path,
    output_dir: Path,
    feature_cols: Optional[Sequence[str]],
    date_col: str,
    code_col: str,
    label_cols: Sequence[str],
    feature_prefix: str,
    mad_multiplier: float,
    output_suffix: str,
    start_date: Optional[int],
    end_date: Optional[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = list_daily_parquet(input_dir, start_date, end_date)
    if not paths:
        raise FileNotFoundError(f"no daily parquet files found in {input_dir}")

    frames = []
    for path in paths:
        df = pd.read_parquet(path)
        if date_col not in df.columns:
            df[date_col] = int(path.stem)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    if code_col not in full.columns:
        raise KeyError(f"code column not found: {code_col}")

    if feature_cols is None:
        selected = select_numeric_columns(full, [date_col, code_col, *label_cols], feature_prefix=feature_prefix)
    else:
        selected = list(feature_cols)
    out = mad_standardize_frame(
        df=full,
        group_col=code_col,
        feature_cols=selected,
        mad_multiplier=mad_multiplier,
        output_suffix=output_suffix,
    )

    day_values = out[date_col]
    if pd.api.types.is_datetime64_any_dtype(day_values):
        day_names = pd.to_datetime(day_values).dt.strftime("%Y%m%d")
    else:
        day_names = day_values.astype(str).str.replace(r"\D", "", regex=True).str[:8]

    out = out.assign(_day_name=day_names)
    for day, day_df in out.groupby("_day_name", sort=True):
        if not str(day).isdigit() or len(str(day)) != 8:
            continue
        day_df.drop(columns=["_day_name"]).to_parquet(output_dir / f"{day}.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cross_section", "time_series"], required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("data/data_train/alpha_360F_day"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=str, default=None)
    parser.add_argument("--feature-prefix", type=str, default="")
    parser.add_argument("--date-col", type=str, default="Date")
    parser.add_argument("--code-col", type=str, default="Code")
    parser.add_argument("--label-cols", type=str, default="")
    parser.add_argument("--mad-multiplier", type=float, default=3.0)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--start-date", type=int, default=None)
    parser.add_argument("--end-date", type=int, default=None)
    args = parser.parse_args()

    feature_cols = parse_columns(args.columns)
    label_cols = parse_columns(args.label_cols) or []
    if args.mode == "cross_section":
        cross_section_standardize_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            feature_cols=feature_cols,
            date_col=args.date_col,
            code_col=args.code_col,
            label_cols=label_cols,
            feature_prefix=args.feature_prefix,
            mad_multiplier=args.mad_multiplier,
            output_suffix=args.output_suffix,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    else:
        time_series_standardize_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            feature_cols=feature_cols,
            date_col=args.date_col,
            code_col=args.code_col,
            label_cols=label_cols,
            feature_prefix=args.feature_prefix,
            mad_multiplier=args.mad_multiplier,
            output_suffix=args.output_suffix,
            start_date=args.start_date,
            end_date=args.end_date,
        )


if __name__ == "__main__":
    main()
