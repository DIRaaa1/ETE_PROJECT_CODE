import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from .io_utils import collect_backtest_frame, read_json, resolve_path, write_json
    from .metrics import compute_cumulative_pnl, compute_daily_stats, compute_outliers, compute_summary_stats, pivot_daily_stats
    from .report import generate_report
except ImportError:
    from io_utils import collect_backtest_frame, read_json, resolve_path, write_json
    from metrics import compute_cumulative_pnl, compute_daily_stats, compute_outliers, compute_summary_stats, pivot_daily_stats
    from report import generate_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_ROOT = Path(__file__).resolve().parent


def default_config_path() -> Path:
    cwd_config = Path.cwd() / "backtest" / "BacktestHyperParams.json"
    if cwd_config.exists():
        return cwd_config.resolve()
    return (BACKTEST_ROOT / "BacktestHyperParams.json").resolve()


def parse_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return items or None


def parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    items = parse_list(value)
    if not items:
        return None
    return [float(x) for x in items]


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(config)
    fields = [
        "prediction_path",
        "target_path",
        "target_dir",
        "label_dir",
        "target_file_pattern",
        "bet_size_path",
        "bet_size_dir",
        "bet_size_file_pattern",
        "output_root",
        "run_name",
        "start_date",
        "end_date",
        "position_method",
    ]
    for field in fields:
        value = getattr(args, field, None)
        if value is not None:
            out[field] = value

    for field in ["signal_cols", "target_cols", "bet_size_cols"]:
        value = parse_list(getattr(args, field, None))
        if value is not None:
            out[field] = value

    quantiles = parse_float_list(getattr(args, "quantiles", None))
    if quantiles is not None:
        out["quantiles"] = quantiles

    if args.transaction_cost_bps is not None:
        out["transaction_cost_bps"] = float(args.transaction_cost_bps)
    if args.default_bet_size is not None:
        out["default_bet_size"] = float(args.default_bet_size)
    if args.make_report is not None:
        out["make_report"] = bool(args.make_report)
    if args.save_positions is not None:
        out["save_positions"] = bool(args.save_positions)
    if args.save_merged_input is not None:
        out["save_merged_input"] = bool(args.save_merged_input)
    return out


def prepare_output_dirs(config: Dict[str, Any], base_dir: str) -> Dict[str, str]:
    output_root = resolve_path(config.get("output_root", "outputs"), base_dir)
    assert output_root is not None
    run_name = config.get("run_name")
    if not run_name:
        run_name = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = os.path.join(output_root, str(run_name))
    dirs = {
        "run_root": run_root,
        "raw": os.path.join(run_root, "RAW_DATA"),
        "daily": os.path.join(run_root, "DAILY_STATS"),
        "summary": os.path.join(run_root, "SUMMARY_STATS"),
        "outliers": os.path.join(run_root, "OUTLIERS"),
        "cumulative": os.path.join(run_root, "CUMULATIVE_PNL"),
        "report": os.path.join(run_root, "REPORT"),
        "config": os.path.join(run_root, "RUN_CONFIG"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def save_frame(frame: pd.DataFrame, csv_path: str, pkl_path: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if pkl_path:
        frame.to_pickle(pkl_path)


def save_daily_split(daily_stats: pd.DataFrame, daily_dir: str) -> None:
    split_dir = os.path.join(daily_dir, "by_date")
    os.makedirs(split_dir, exist_ok=True)
    for date_value, g in daily_stats.groupby("date", sort=True):
        g.to_csv(os.path.join(split_dir, f"{date_value}.csv"), index=False, encoding="utf-8-sig")


def build_run_metadata(
    config: Dict[str, Any],
    input_frame: pd.DataFrame,
    signal_cols: List[str],
    target_cols: List[str],
    bet_cols: List[str],
    prediction_files: List[str],
) -> Dict[str, Any]:
    date_col = str(config.get("date_col", "Date"))
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_rows": int(len(input_frame)),
        "input_dates": int(input_frame[date_col].nunique()) if date_col in input_frame.columns else 0,
        "signal_cols": signal_cols,
        "target_cols": target_cols,
        "bet_size_cols": bet_cols,
        "prediction_files": prediction_files,
    }


def run_backtest(config: Dict[str, Any], config_base_dir: str) -> str:
    date_col = str(config.get("date_col", "Date"))
    dirs = prepare_output_dirs(config, config_base_dir)

    input_frame, signal_cols, target_cols, bet_cols, prediction_files = collect_backtest_frame(config, config_base_dir)
    if input_frame.empty:
        raise RuntimeError("The merged backtest input is empty")

    config = dict(config)
    metadata = build_run_metadata(config, input_frame, signal_cols, target_cols, bet_cols, prediction_files)
    config.update(metadata)

    write_json(config, os.path.join(dirs["config"], "BacktestHyperParams.json"))
    pd.DataFrame({"prediction_file": prediction_files}).to_csv(
        os.path.join(dirs["config"], "prediction_files.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    preview_cols = [date_col, str(config.get("symbol_col", "Code"))] + signal_cols + target_cols + bet_cols
    preview_cols = [c for c in preview_cols if c in input_frame.columns]
    input_frame[preview_cols].head(int(config.get("input_preview_rows", 1000))).to_csv(
        os.path.join(dirs["raw"], "input_preview.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    if bool(config.get("save_merged_input", False)):
        input_frame.to_csv(os.path.join(dirs["raw"], "merged_input.csv"), index=False, encoding="utf-8-sig")

    daily_stats, positions = compute_daily_stats(input_frame, date_col, signal_cols, target_cols, bet_cols, config)
    daily_wide = pivot_daily_stats(daily_stats)
    summary_stats = compute_summary_stats(daily_stats, float(config.get("annualization", 252.0)))
    summary_wide = summary_stats.pivot_table(
        index=["signal", "target", "qrank", "bet_size_col"],
        columns="stat_type",
        values="value",
        aggfunc="first",
    ).reset_index() if not summary_stats.empty else pd.DataFrame()
    if not summary_wide.empty:
        summary_wide.columns.name = None
    cumulative_pnl = compute_cumulative_pnl(daily_stats)
    outliers = compute_outliers(
        daily_stats,
        config.get("outlier_metrics", ["pnl", "ppd", "ic", "rank_ic"]),
        int(config.get("outlier_top_k", 10)),
    )

    save_frame(daily_stats, os.path.join(dirs["daily"], "daily_stats.csv"), os.path.join(dirs["daily"], "daily_stats.pkl"))
    save_frame(daily_wide, os.path.join(dirs["daily"], "daily_stats_wide.csv"))
    if bool(config.get("save_daily_files", True)):
        save_daily_split(daily_stats, dirs["daily"])
    save_frame(summary_stats, os.path.join(dirs["summary"], "summary_stats.csv"), os.path.join(dirs["summary"], "summary_stats.pkl"))
    if not summary_wide.empty:
        save_frame(summary_wide, os.path.join(dirs["summary"], "summary_stats_wide.csv"))
    save_frame(cumulative_pnl, os.path.join(dirs["cumulative"], "cumulative_pnl.csv"))
    save_frame(outliers, os.path.join(dirs["outliers"], "outliers.csv"))

    if bool(config.get("save_positions", False)) and not positions.empty:
        save_frame(positions, os.path.join(dirs["daily"], "positions.csv"))

    if bool(config.get("make_report", True)):
        report_path = generate_report(
            os.path.join(dirs["report"], "backtest_report.pdf"),
            summary_stats,
            cumulative_pnl,
            config,
        )
        config["report_path"] = report_path
        write_json(config, os.path.join(dirs["config"], "BacktestHyperParams.json"))

    print(f"Backtest finished: {dirs['run_root']}")
    print(f"Daily stats: {os.path.join(dirs['daily'], 'daily_stats.csv')}")
    print(f"Summary stats: {os.path.join(dirs['summary'], 'summary_stats.csv')}")
    if bool(config.get("make_report", True)):
        print(f"Report: {config.get('report_path')}")
    return dirs["run_root"]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path).resolve() if config_path else default_config_path()
    return read_json(str(path))


def run_backtest_from_config(config_path: Optional[str] = None, **overrides: Any) -> str:
    path = Path(config_path).resolve() if config_path else default_config_path()
    config = read_json(str(path))
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return run_backtest(config, str(path.parent))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path()))
    parser.add_argument("--prediction_path", default=None)
    parser.add_argument("--target_path", default=None)
    parser.add_argument("--target_dir", default=None)
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--target_file_pattern", default=None)
    parser.add_argument("--bet_size_path", default=None)
    parser.add_argument("--bet_size_dir", default=None)
    parser.add_argument("--bet_size_file_pattern", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--end_date", default=None)
    parser.add_argument("--signal_cols", default=None)
    parser.add_argument("--target_cols", default=None)
    parser.add_argument("--bet_size_cols", default=None)
    parser.add_argument("--quantiles", default=None)
    parser.add_argument("--position_method", default=None, choices=["sign", "score", "rank"])
    parser.add_argument("--transaction_cost_bps", type=float, default=None)
    parser.add_argument("--default_bet_size", type=float, default=None)
    parser.add_argument("--make_report", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_positions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save_merged_input", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> str:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = read_json(str(config_path))
    config = apply_overrides(config, args)
    return run_backtest(config, str(config_path.parent))


if __name__ == "__main__":
    main()
