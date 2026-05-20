import os
import re
import numpy as np
import pandas as pd


MAXIMIZE_METRICS = {
    "DailyICMean",
    "DailyTop5ReturnMean",
    "DailyTop10ReturnMean",
    "DailyTop20ReturnMean",
    "DailyTop5MinusBottom5Mean",
    "DailyTop10MinusBottom10Mean",
    "DailyTop20MinusBottom10Mean",
    "DailyTop20MinusBottom20Mean",
    "DailyTop5ExcessMarketMean",
    "DailyTop10ExcessMarketMean",
    "DailyTop20ExcessMarketMean",
}

MINIMIZE_METRICS = {
    "DailyMSEMean",
}


def _safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def _safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_mse(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return 0.0
    return float(np.mean((x - y) ** 2))


def _pick_k(n, pct):
    return max(int(np.ceil(n * pct)), 1)


def _resolve_rank_score(df, rank_by, pred_col, label_col, rank_abs):
    if rank_by == "pred":
        score = pd.to_numeric(df[pred_col], errors="coerce")
    elif rank_by == "label":
        score = pd.to_numeric(df[label_col], errors="coerce")
    else:
        raise ValueError(f"Unknown rank_by={rank_by}")
    if rank_abs:
        score = score.abs()
    return score


def normalize_numeric_code_series(s):
    s = pd.Series(s).astype(str).str.strip()
    out = s.str.extract(r"(\d{6})", expand=False)
    miss = out.isna()
    if miss.any():
        digits = s[miss].str.replace(r"\D", "", regex=True).str.slice(0, 6)
        digits = digits.where(digits.str.len() >= 6)
        out.loc[miss] = digits
    return out.where(out.notna(), None)


def normalize_date_series(s):
    s = pd.Series(s)
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s.dt.strftime("%Y%m%d")
    else:
        out = s.astype(str).str.strip().str.replace(r"\D", "", regex=True).str.slice(0, 8)
    out = out.where(out.str.len() >= 8)
    return pd.to_numeric(out, errors="coerce").astype("Int64")


def to_bool_safe_series(s):
    s = pd.Series(s)
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        out = pd.Series(False, index=s.index, dtype=bool)
        mask = s.notna()
        out.loc[mask] = s.loc[mask].astype(float) != 0
        return out

    s2 = s.astype(str).str.strip().str.lower()
    true_set = {"1", "true", "t", "y", "yes"}
    false_set = {"0", "false", "f", "n", "no", "nan", "none", ""}
    out = pd.Series(False, index=s.index, dtype=bool)
    out.loc[s2.isin(true_set)] = True
    out.loc[s2.isin(false_set)] = False
    return out


def build_file_map(directory, suffixes=(".parquet", ".pqt")):
    file_map = {}
    if directory is None or str(directory).strip() == "" or not os.path.isdir(directory):
        return file_map

    suffixes = {str(x).lower() for x in suffixes}
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        suffix = os.path.splitext(name)[1].lower()
        if suffix not in suffixes:
            continue
        m = re.search(r"\d{8}", os.path.splitext(name)[0])
        if m:
            d = int(m.group(0))
            if d not in file_map:
                file_map[d] = path
    return file_map


def load_daily_universe_codes(
    universe_path,
    include_paused=True,
    include_st=True,
    include_open_high_limit=True,
    include_open_low_limit=True,
    code_col="Code",
):
    uni = pd.read_parquet(universe_path)
    required_cols = [code_col]

    filter_cols = [
        ("paused", include_paused),
        ("st", include_st),
        ("open_high_limit", include_open_high_limit),
        ("open_low_limit", include_open_low_limit),
    ]
    required_cols.extend([col for col, include_flag in filter_cols if not include_flag])

    missing_cols = [c for c in required_cols if c not in uni.columns]
    if missing_cols:
        raise KeyError(f"daily universe Invalid {missing_cols}: {universe_path}")

    uni[code_col] = normalize_numeric_code_series(uni[code_col])
    uni = uni.dropna(subset=[code_col]).copy()

    for col, _ in filter_cols:
        if col in uni.columns:
            uni[col] = to_bool_safe_series(uni[col])

    if not include_paused:
        uni = uni.loc[~uni["paused"]].copy()
    if not include_st:
        uni = uni.loc[~uni["st"]].copy()
    if not include_open_high_limit:
        uni = uni.loc[~uni["open_high_limit"]].copy()
    if not include_open_low_limit:
        uni = uni.loc[~uni["open_low_limit"]].copy()

    uni = uni.drop_duplicates(subset=[code_col], keep="first")
    return set(uni[code_col].tolist())


def build_daily_universe_code_map(
    universe_dir,
    candidate_dates,
    include_paused=True,
    include_st=True,
    include_open_high_limit=True,
    include_open_low_limit=True,
    strict_missing=False,
):
    universe_files = build_file_map(universe_dir, suffixes=(".parquet", ".pqt"))
    candidate_dates = sorted({int(d) for d in candidate_dates if pd.notna(d)})

    missing_dates = [d for d in candidate_dates if d not in universe_files]
    if strict_missing and missing_dates:
        raise FileNotFoundError(
            f"daily universe Invalid {len(missing_dates)} Invalid Invalid 10 Invalid: {missing_dates[:10]}"
        )

    code_map = {}
    for d in candidate_dates:
        path = universe_files.get(d)
        if path is None:
            continue
        code_map[d] = load_daily_universe_codes(
            path,
            include_paused=include_paused,
            include_st=include_st,
            include_open_high_limit=include_open_high_limit,
            include_open_low_limit=include_open_low_limit,
        )

    return code_map, {
        "candidate_dates": len(candidate_dates),
        "available_dates": len(code_map),
        "missing_dates": missing_dates,
    }


def filter_monitor_frame_by_daily_universe(
    df,
    date_col,
    symbol_col,
    universe_dir=None,
    universe_code_map=None,
    include_paused=True,
    include_st=True,
    include_open_high_limit=True,
    include_open_low_limit=True,
    strict_missing=False,
):
    required_cols = [date_col, symbol_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"daily universe Invalid: {missing_cols}")

    norm_dates = normalize_date_series(df[date_col])
    norm_symbols = normalize_numeric_code_series(df[symbol_col])
    candidate_dates = sorted(norm_dates.dropna().astype(int).unique().tolist())

    if universe_code_map is None:
        universe_code_map, build_stats = build_daily_universe_code_map(
            universe_dir=universe_dir,
            candidate_dates=candidate_dates,
            include_paused=include_paused,
            include_st=include_st,
            include_open_high_limit=include_open_high_limit,
            include_open_low_limit=include_open_low_limit,
            strict_missing=strict_missing,
        )
    else:
        build_stats = {
            "candidate_dates": len(candidate_dates),
            "available_dates": len(universe_code_map),
            "missing_dates": [d for d in candidate_dates if d not in universe_code_map],
        }
        if strict_missing and build_stats["missing_dates"]:
            raise FileNotFoundError(
                f"daily universe Invalid {len(build_stats['missing_dates'])} Invalid "
                f"Invalid 10 Invalid: {build_stats['missing_dates'][:10]}"
            )

    keep = pd.Series(False, index=df.index, dtype=bool)
    for d, codes in universe_code_map.items():
        if not codes:
            continue
        date_mask = norm_dates == int(d)
        if date_mask.any():
            keep.loc[date_mask] = norm_symbols.loc[date_mask].isin(codes)

    filtered = df.loc[keep.to_numpy()].copy()
    output_dates = normalize_date_series(filtered[date_col]).dropna().astype(int).nunique() if len(filtered) else 0

    stats = {
        "input_rows": int(len(df)),
        "output_rows": int(len(filtered)),
        "dropped_rows": int(len(df) - len(filtered)),
        "input_dates": int(len(candidate_dates)),
        "output_dates": int(output_dates),
        "universe_available_dates": int(build_stats.get("available_dates", len(universe_code_map))),
        "universe_missing_dates": build_stats.get("missing_dates", []),
        "include_paused": bool(include_paused),
        "include_st": bool(include_st),
        "include_open_high_limit": bool(include_open_high_limit),
        "include_open_low_limit": bool(include_open_low_limit),
    }
    return filtered, stats


def _portfolio_return(df, pct, side, pred_col, label_col, return_col, rank_by, rank_abs):
    work = df[[pred_col, label_col, return_col]].copy()
    work[return_col] = pd.to_numeric(work[return_col], errors="coerce")
    work["__rank_score__"] = _resolve_rank_score(work, rank_by, pred_col, label_col, rank_abs)
    work = work[np.isfinite(work["__rank_score__"]) & np.isfinite(work[return_col])]
    if work.empty:
        return 0.0

    k = _pick_k(len(work), pct)
    if side == "top":
        work = work.sort_values("__rank_score__", ascending=False)
    elif side == "bottom":
        work = work.sort_values("__rank_score__", ascending=True)
    else:
        raise ValueError(f"Unknown side={side}")

    return float(work[return_col].iloc[:k].mean())


def build_monitor_frame(date_series, symbol_series, pred_array, label_array, return_array, date_col, symbol_col):
    return pd.DataFrame({
        date_col: pd.Series(date_series).astype(str).values,
        symbol_col: pd.Series(symbol_series).astype(str).values,
        "y_pred": np.asarray(pred_array).reshape(-1),
        "y_true": np.asarray(label_array).reshape(-1),
        "monitor_return": np.asarray(return_array).reshape(-1),
    })


def compute_monitor_metrics(
    df,
    date_col,
    pred_col="y_pred",
    label_col="y_true",
    return_col="monitor_return",
    rank_by="pred",
    rank_abs=False
):
    required_cols = [date_col, pred_col, label_col, return_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Invalid: {missing_cols}")

    daily_ic = []
    daily_mse = []
    daily_top5 = []
    daily_top10 = []
    daily_top20 = []
    daily_top5_minus_bottom5 = []
    daily_top10_minus_bottom10 = []
    daily_top20_minus_bottom10 = []
    daily_top20_minus_bottom20 = []
    daily_top5_excess_market = []
    daily_top10_excess_market = []
    daily_top20_excess_market = []

    for _, g in df.groupby(date_col, sort=True):
        pred = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
        label = pd.to_numeric(g[label_col], errors="coerce").to_numpy()
        ret = pd.to_numeric(g[return_col], errors="coerce")
        ret_np = ret.to_numpy()

        ic_val = _safe_corr(pred, label)
        mse_val = _safe_mse(pred, label)
        top5_val = _portfolio_return(
            g, 0.05, "top", pred_col, label_col, return_col, rank_by, rank_abs
        )
        top10_val = _portfolio_return(
            g, 0.10, "top", pred_col, label_col, return_col, rank_by, rank_abs
        )
        top20_val = _portfolio_return(
            g, 0.20, "top", pred_col, label_col, return_col, rank_by, rank_abs
        )
        bottom5_val = _portfolio_return(
            g, 0.05, "bottom", pred_col, label_col, return_col, rank_by, rank_abs
        )
        bottom10_val = _portfolio_return(
            g, 0.10, "bottom", pred_col, label_col, return_col, rank_by, rank_abs
        )
        bottom20_val = _portfolio_return(
            g, 0.20, "bottom", pred_col, label_col, return_col, rank_by, rank_abs
        )

        market_mask = np.isfinite(ret_np)
        market_val = float(np.mean(ret_np[market_mask])) if market_mask.sum() > 0 else 0.0

        daily_ic.append(ic_val)
        daily_mse.append(mse_val)
        daily_top5.append(top5_val)
        daily_top10.append(top10_val)
        daily_top20.append(top20_val)
        daily_top5_minus_bottom5.append(top5_val - bottom5_val)
        daily_top10_minus_bottom10.append(top10_val - bottom10_val)
        daily_top20_minus_bottom10.append(top20_val - bottom10_val)
        daily_top20_minus_bottom20.append(top20_val - bottom20_val)
        daily_top5_excess_market.append(top5_val - market_val)
        daily_top10_excess_market.append(top10_val - market_val)
        daily_top20_excess_market.append(top20_val - market_val)

    return {
        "DailyICMean": _safe_mean(daily_ic),
        "DailyMSEMean": _safe_mean(daily_mse),
        "DailyTop5ReturnMean": _safe_mean(daily_top5),
        "DailyTop10ReturnMean": _safe_mean(daily_top10),
        "DailyTop20ReturnMean": _safe_mean(daily_top20),
        "DailyTop5MinusBottom5Mean": _safe_mean(daily_top5_minus_bottom5),
        "DailyTop10MinusBottom10Mean": _safe_mean(daily_top10_minus_bottom10),
        "DailyTop20MinusBottom10Mean": _safe_mean(daily_top20_minus_bottom10),
        "DailyTop20MinusBottom20Mean": _safe_mean(daily_top20_minus_bottom20),
        "DailyTop5ExcessMarketMean": _safe_mean(daily_top5_excess_market),
        "DailyTop10ExcessMarketMean": _safe_mean(daily_top10_excess_market),
        "DailyTop20ExcessMarketMean": _safe_mean(daily_top20_excess_market),
        "NumDates": int(df[date_col].nunique()),
    }


def get_metric_mode(metric_name, explicit_mode="auto"):
    if explicit_mode in {"max", "min"}:
        return explicit_mode
    if metric_name in MAXIMIZE_METRICS:
        return "max"
    if metric_name in MINIMIZE_METRICS:
        return "min"
    raise ValueError(f"Unknown metric_name={metric_name}")


def metric_value_is_better(current_value, best_value, mode):
    if current_value is None or not np.isfinite(current_value):
        return False
    if best_value is None or not np.isfinite(best_value):
        return True
    if mode == "max":
        return current_value > best_value
    if mode == "min":
        return current_value < best_value
    raise ValueError(f"Unknown mode={mode}")


def metric_sort_value(metric_value, mode):
    if metric_value is None or not np.isfinite(metric_value):
        return -np.inf if mode == "max" else np.inf
    return float(metric_value)


def sort_model_records(model_records, mode):
    reverse = mode == "max"
    return sorted(model_records, key=lambda x: metric_sort_value(x[0], mode), reverse=reverse)


def save_monitor_history(history_rows, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame(history_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")