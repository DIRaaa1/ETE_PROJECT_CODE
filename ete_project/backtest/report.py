import os
from typing import Any, Dict

import pandas as pd


def metric_frame(summary_stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary_stats.empty:
        return pd.DataFrame()
    return summary_stats[summary_stats["stat_type"] == metric].copy()


def wide_metric(summary_stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    frame = metric_frame(summary_stats, metric)
    if frame.empty:
        return pd.DataFrame()
    return frame.pivot_table(
        index=["signal", "target", "bet_size_col"],
        columns="qrank",
        values="value",
        aggfunc="first",
    ).reset_index()


def generate_report(
    report_path: str,
    summary_stats: pd.DataFrame,
    cumulative_pnl: pd.DataFrame,
    config: Dict[str, Any],
) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        skipped = os.path.join(os.path.dirname(report_path), "report_skipped.txt")
        with open(skipped, "w", encoding="utf-8") as f:
            f.write("matplotlib is not available; report generation was skipped.\n")
        return skipped

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Backtest Summary", fontsize=16)
        ax = fig.add_subplot(111)
        ax.axis("off")
        lines = [
            f"Rows: {int(config.get('input_rows', 0))}",
            f"Dates: {int(config.get('input_dates', 0))}",
            f"Signals: {', '.join(config.get('signal_cols', []))}",
            f"Targets: {', '.join(config.get('target_cols', []))}",
            f"Quantiles: {', '.join(str(x) for x in config.get('quantiles', []))}",
            f"Position method: {config.get('position_method', 'sign')}",
            f"Transaction cost bps: {config.get('transaction_cost_bps', 0.0)}",
        ]
        ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for metric in ["sharpe", "ppd", "hit_ratio", "ic", "rank_ic"]:
            frame = wide_metric(summary_stats, metric)
            if frame.empty:
                continue
            value_cols = [c for c in frame.columns if str(c).startswith("qr_")]
            if not value_cols:
                continue
            labels = frame[["signal", "target"]].astype(str).agg(" / ".join, axis=1)
            plot_frame = frame[value_cols].copy()
            plot_frame.index = labels
            if len(plot_frame) > 20:
                plot_frame = plot_frame.head(20)
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_frame.plot(kind="bar", ax=ax)
            ax.set_title(metric)
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", labelrotation=45)
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if not cumulative_pnl.empty:
            report_qrank = str(config.get("report_qrank", "qr_25"))
            frame = cumulative_pnl[cumulative_pnl["qrank"] == report_qrank].copy()
            if frame.empty:
                frame = cumulative_pnl.copy()
            frame["date_dt"] = pd.to_datetime(frame["date"].astype(str), format="%Y%m%d", errors="coerce")
            for target, g in frame.groupby("target", sort=True):
                fig, ax = plt.subplots(figsize=(12, 6))
                for key_values, line in g.groupby(["signal", "qrank", "bet_size_col"], sort=True):
                    label = " / ".join(str(x) for x in key_values)
                    line = line.sort_values("date")
                    ax.plot(line["date_dt"], line["cum_pnl"], label=label)
                ax.set_title(f"Cumulative PnL: {target}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Cumulative PnL")
                tick_dates = (
                    g[["date_dt"]]
                    .dropna()
                    .drop_duplicates()
                    .sort_values("date_dt")["date_dt"]
                )
                if not tick_dates.empty:
                    ticks = tick_dates[
                        (tick_dates.dt.month.isin([1, 7]))
                        & (tick_dates.dt.day <= 7)
                    ]
                    if ticks.empty:
                        ticks = tick_dates.iloc[::max(int(len(tick_dates) / 8), 1)]
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(ticks.dt.strftime("%Y-%m"))
                ax.tick_params(axis="x", labelrotation=45)
                ax.grid(alpha=0.3)
                if len(g[["signal", "qrank", "bet_size_col"]].drop_duplicates()) <= 12:
                    ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    return report_path
