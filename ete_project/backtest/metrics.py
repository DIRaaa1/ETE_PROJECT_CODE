import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def safe_mean(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def safe_sum(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.sum())


def safe_std(values: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1))


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return 0.0
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def safe_rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    a = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    b = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return safe_corr(a, b)


def quantile_label(value: float) -> str:
    pct = int(round(float(value) * 100))
    return f"qr_{pct}"


def pick_count(n: int, fraction: float) -> int:
    if n <= 0:
        return 0
    return max(int(math.ceil(n * float(fraction))), 1)


def select_quantile(work: pd.DataFrame, q: float, rank_by_abs: bool) -> pd.DataFrame:
    if work.empty:
        return work.copy()
    score = work["signal"].abs() if rank_by_abs else work["signal"]
    order = score.sort_values(ascending=False).index
    k = pick_count(len(order), q)
    return work.loc[order[:k]].copy()


def build_positions(signal: pd.Series, method: str, leverage_cap: float) -> pd.Series:
    if method == "score":
        pos = signal.astype(float)
    elif method == "rank":
        ranks = signal.rank(method="average", pct=True).astype(float)
        pos = 2.0 * ranks - 1.0
    elif method == "sign":
        pos = np.sign(signal.astype(float))
    else:
        raise ValueError(f"Unsupported position_method={method}")
    pos = pd.Series(pos, index=signal.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if leverage_cap and float(leverage_cap) > 0:
        cap = float(leverage_cap)
        pos = pos.clip(lower=-cap, upper=cap)
    return pos


def daily_one_combo(
    df: pd.DataFrame,
    date_value: str,
    signal_col: str,
    target_col: str,
    bet_col: str,
    q: float,
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    work = pd.DataFrame({
        "signal": pd.to_numeric(df[signal_col], errors="coerce"),
        "target": pd.to_numeric(df[target_col], errors="coerce"),
        "bet": pd.to_numeric(df[bet_col], errors="coerce"),
    })
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if bool(config.get("nonzero_signals_only", True)):
        work = work[work["signal"] != 0]
    work = work[work["bet"] > 0]
    selected = select_quantile(work, q, bool(config.get("rank_by_abs", True)))

    position_method = str(config.get("position_method", "sign"))
    leverage_cap = float(config.get("leverage_cap", 1.0))
    cost_bps = float(config.get("transaction_cost_bps", 0.0))

    if selected.empty:
        base = {
            "date": date_value,
            "signal": signal_col,
            "target": target_col,
            "qrank": quantile_label(q),
            "bet_size_col": bet_col,
        }
        rows = []
        for stat in [
            "pnl",
            "ppd",
            "sizeNotional",
            "n_trades",
            "nrInstr",
            "hit_ratio",
            "long_ratio",
            "short_ratio",
            "ic",
            "rank_ic",
            "avg_signal",
            "avg_abs_signal",
        ]:
            item = dict(base)
            item["stat_type"] = stat
            item["value"] = 0.0
            rows.append(item)
        return rows, pd.DataFrame()

    positions = build_positions(selected["signal"], position_method, leverage_cap)
    gross_pnl = positions * selected["target"] * selected["bet"]
    cost = positions.abs() * selected["bet"] * cost_bps / 10000.0
    pnl = gross_pnl - cost
    size_notional = float((positions.abs() * selected["bet"]).sum())
    ppd = float(pnl.sum() / size_notional) if size_notional > 0 else 0.0
    traded = positions != 0
    n_trades = int(traded.sum())

    signed_return = positions * selected["target"]
    hit_ratio = float((signed_return[traded] > 0).mean()) if n_trades > 0 else 0.0
    long_ratio = float((positions[traded] > 0).mean()) if n_trades > 0 else 0.0
    short_ratio = float((positions[traded] < 0).mean()) if n_trades > 0 else 0.0

    metrics = {
        "pnl": float(pnl.sum()),
        "ppd": ppd,
        "sizeNotional": size_notional,
        "n_trades": float(n_trades),
        "nrInstr": float(n_trades),
        "hit_ratio": hit_ratio,
        "long_ratio": long_ratio,
        "short_ratio": short_ratio,
        "ic": safe_corr(selected["signal"], selected["target"]),
        "rank_ic": safe_rank_corr(selected["signal"], selected["target"]),
        "avg_signal": float(selected["signal"].mean()),
        "avg_abs_signal": float(selected["signal"].abs().mean()),
    }

    base = {
        "date": date_value,
        "signal": signal_col,
        "target": target_col,
        "qrank": quantile_label(q),
        "bet_size_col": bet_col,
    }
    rows = []
    for stat, value in metrics.items():
        item = dict(base)
        item["stat_type"] = stat
        item["value"] = float(value)
        rows.append(item)

    selected_rows = selected.copy()
    selected_rows["date"] = date_value
    selected_rows["signal_name"] = signal_col
    selected_rows["target_name"] = target_col
    selected_rows["qrank"] = quantile_label(q)
    selected_rows["bet_size_col"] = bet_col
    selected_rows["position"] = positions
    selected_rows["gross_pnl"] = gross_pnl
    selected_rows["cost"] = cost
    selected_rows["pnl"] = pnl
    return rows, selected_rows


def compute_daily_stats(
    df: pd.DataFrame,
    date_col: str,
    signal_cols: Sequence[str],
    target_cols: Sequence[str],
    bet_cols: Sequence[str],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quantiles = [float(x) for x in config.get("quantiles", [1.0, 0.75, 0.5, 0.25])]
    rows: List[Dict[str, Any]] = []
    position_frames: List[pd.DataFrame] = []

    for date_value, day_df in df.groupby(date_col, sort=True):
        day = str(date_value)
        for signal_col in signal_cols:
            for target_col in target_cols:
                for bet_col in bet_cols:
                    for q in quantiles:
                        stat_rows, pos_frame = daily_one_combo(day_df, day, signal_col, target_col, bet_col, q, config)
                        rows.extend(stat_rows)
                        if bool(config.get("save_positions", False)) and not pos_frame.empty:
                            position_frames.append(pos_frame)

    daily_stats = pd.DataFrame(rows)
    if position_frames:
        positions = pd.concat(position_frames, ignore_index=True, copy=False)
    else:
        positions = pd.DataFrame()
    return daily_stats, positions


def pivot_daily_stats(daily_stats: pd.DataFrame) -> pd.DataFrame:
    if daily_stats.empty:
        return pd.DataFrame()
    keys = ["date", "signal", "target", "qrank", "bet_size_col"]
    out = daily_stats.pivot_table(index=keys, columns="stat_type", values="value", aggfunc="first").reset_index()
    out.columns.name = None
    return out


def annualized_sharpe(values: Sequence[float], annualization: float) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(arr.mean() / sd * math.sqrt(float(annualization)))


def t_stat(values: Sequence[float]) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    sd = arr.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(arr.mean() / (sd / math.sqrt(arr.size)))


def compute_summary_stats(daily_stats: pd.DataFrame, annualization: float) -> pd.DataFrame:
    wide = pivot_daily_stats(daily_stats)
    if wide.empty:
        return pd.DataFrame(columns=["signal", "target", "qrank", "bet_size_col", "stat_type", "value"])

    keys = ["signal", "target", "qrank", "bet_size_col"]
    rows: List[Dict[str, Any]] = []
    for key_values, g in wide.groupby(keys, sort=True):
        base = dict(zip(keys, key_values))
        pnl = g["pnl"] if "pnl" in g.columns else pd.Series(dtype=float)
        ppd = g["ppd"] if "ppd" in g.columns else pd.Series(dtype=float)
        size = g["sizeNotional"] if "sizeNotional" in g.columns else pd.Series(dtype=float)
        n_trades = g["n_trades"] if "n_trades" in g.columns else pd.Series(dtype=float)

        metric_map = {
            "pnl": safe_sum(pnl),
            "pnl_mean": safe_mean(pnl),
            "pnl_std": safe_std(pnl),
            "sharpe": annualized_sharpe(pnl, annualization),
            "t_stat": t_stat(pnl),
            "ppd": safe_sum(pnl) / safe_sum(size) if safe_sum(size) > 0 else 0.0,
            "daily_ppd_mean": safe_mean(ppd),
            "daily_ppd_median": float(pd.to_numeric(ppd, errors="coerce").dropna().median()) if len(ppd) else 0.0,
            "sizeNotional": safe_sum(size),
            "sizeNotional_mean": safe_mean(size),
            "n_trades": safe_sum(n_trades),
            "n_trades_mean": safe_mean(n_trades),
            "hit_ratio": safe_mean(g["hit_ratio"]) if "hit_ratio" in g.columns else 0.0,
            "long_ratio": safe_mean(g["long_ratio"]) if "long_ratio" in g.columns else 0.0,
            "short_ratio": safe_mean(g["short_ratio"]) if "short_ratio" in g.columns else 0.0,
            "ic": safe_mean(g["ic"]) if "ic" in g.columns else 0.0,
            "rank_ic": safe_mean(g["rank_ic"]) if "rank_ic" in g.columns else 0.0,
            "n_days": float(g["date"].nunique()),
        }

        for stat_type, value in metric_map.items():
            row = dict(base)
            row["stat_type"] = stat_type
            row["value"] = float(value)
            rows.append(row)

    return pd.DataFrame(rows)


def compute_cumulative_pnl(daily_stats: pd.DataFrame) -> pd.DataFrame:
    wide = pivot_daily_stats(daily_stats)
    if wide.empty or "pnl" not in wide.columns:
        return pd.DataFrame(columns=["date", "signal", "target", "qrank", "bet_size_col", "pnl", "cum_pnl"])
    keys = ["signal", "target", "qrank", "bet_size_col"]
    out = wide[["date"] + keys + ["pnl"]].copy()
    out = out.sort_values(["signal", "target", "qrank", "bet_size_col", "date"])
    out["cum_pnl"] = out.groupby(keys, sort=False)["pnl"].cumsum()
    return out


def compute_outliers(daily_stats: pd.DataFrame, metrics: Sequence[str], top_k: int) -> pd.DataFrame:
    if daily_stats.empty or top_k <= 0:
        return pd.DataFrame()
    keys = ["signal", "target", "qrank", "bet_size_col", "stat_type"]
    rows: List[Dict[str, Any]] = []
    work = daily_stats[daily_stats["stat_type"].isin(metrics)].copy()
    for key_values, g in work.groupby(keys, sort=True):
        for side, ascending in [("bottom", True), ("top", False)]:
            ordered = g.sort_values("value", ascending=ascending).head(top_k)
            for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
                item = {k: v for k, v in zip(keys, key_values)}
                item["date"] = row["date"]
                item["side"] = side
                item["rank"] = rank
                item["value"] = float(row["value"])
                rows.append(item)
    return pd.DataFrame(rows)
