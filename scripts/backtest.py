from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cross_sectional_ml.backtest import (
    EIGHT_MODELS,
    align_eight_model_predictions,
    attach_outcomes,
    average_daily_spearman,
    backtest_grid,
    daily_predictive_metrics,
    portfolio_summary,
    predictive_summary,
    rank_zscore_ensemble,
    read_model_predictions,
)
from cross_sectional_ml.utils import ProjectPaths, read_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate eight models and their rank ensemble.")
    parser.add_argument("--paths", type=Path, help="Optional project path configuration.")
    parser.add_argument("--protocol", choices=("o2o", "de"))
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Root containing F1/F2 outputs or one <model>.parquet file per model.",
    )
    parser.add_argument("--outcomes", type=Path, help="Optional realised-return table.")
    parser.add_argument("--outcome-column", default="raw_return")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.load(args.paths) if args.paths else None
    if paths and not args.protocol and (args.predictions is None or args.output is None):
        raise ValueError("--protocol is required when --paths supplies backtest locations")
    prediction_root = args.predictions or (
        paths.prediction_root(args.protocol) if paths and args.protocol else None
    )
    output = args.output or (paths.backtest(args.protocol) if paths and args.protocol else None)
    if prediction_root is None or output is None:
        raise ValueError("provide --predictions and --output, or derive them with --paths and --protocol")

    prediction_frames = {
        model: read_model_predictions(prediction_root, model) for model in EIGHT_MODELS
    }
    aligned = align_eight_model_predictions(prediction_frames)
    if args.outcomes:
        outcomes = read_frame(args.outcomes)
    else:
        first = prediction_frames[EIGHT_MODELS[0]]
        if args.outcome_column not in first.columns:
            raise ValueError(
                f"provide --outcomes or prediction files containing {args.outcome_column!r}"
            )
        outcomes = first[["Date", "Code", args.outcome_column]]

    output.mkdir(parents=True, exist_ok=True)
    average_daily_spearman(aligned).to_csv(output / "prediction_correlation.csv", index=True)
    ensemble = rank_zscore_ensemble(aligned).rename(columns={"ensemble_score": "prediction"})
    ensemble.to_parquet(output / "ensemble_predictions.parquet", index=False)

    daily_portfolios: list[pd.DataFrame] = []
    daily_predictive: list[pd.DataFrame] = []
    score_panels = {
        model: aligned[["Date", "Code", model]].rename(columns={model: "score"})
        for model in EIGHT_MODELS
    }
    score_panels["ensemble"] = ensemble.rename(columns={"prediction": "score"})
    for model, scores in score_panels.items():
        panel = attach_outcomes(scores, outcomes, outcome_column=args.outcome_column)
        portfolio = backtest_grid(panel, outcome_column=args.outcome_column)
        portfolio.insert(0, "model", model)
        daily_portfolios.append(portfolio)
        daily_predictive.append(
            daily_predictive_metrics(panel, model=model, outcome_column=args.outcome_column)
        )

    portfolio_daily = pd.concat(daily_portfolios, ignore_index=True)
    predictive_daily = pd.concat(daily_predictive, ignore_index=True)
    portfolio_daily.to_parquet(output / "daily_portfolios.parquet", index=False)
    predictive_daily.to_csv(output / "daily_predictive_metrics.csv", index=False)

    portfolio_tables = [portfolio_summary(portfolio_daily, "2024-2025")]
    predictive_tables = [predictive_summary(predictive_daily, "2024-2025")]
    portfolio_years = pd.to_datetime(portfolio_daily["Date"]).dt.year
    predictive_years = pd.to_datetime(predictive_daily["Date"]).dt.year
    for year in (2024, 2025):
        portfolio_tables.append(portfolio_summary(portfolio_daily.loc[portfolio_years.eq(year)], str(year)))
        predictive_tables.append(
            predictive_summary(predictive_daily.loc[predictive_years.eq(year)], str(year))
        )
    pd.concat(portfolio_tables, ignore_index=True).to_csv(
        output / "portfolio_summary.csv", index=False
    )
    pd.concat(predictive_tables, ignore_index=True).to_csv(
        output / "predictive_summary.csv", index=False
    )
    print(f"evaluated eight models and ensemble over {aligned['Date'].nunique():,} dates")


if __name__ == "__main__":
    main()
