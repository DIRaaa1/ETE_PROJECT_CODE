from __future__ import annotations

import argparse
from pathlib import Path

from cross_sectional_ml.features import load_feature_manifest
from cross_sectional_ml.io import normalize_date_series
from cross_sectional_ml.labels import get_label_spec
from cross_sectional_ml.models import SEQUENCE_LENGTH
from cross_sectional_ml.universe import (
    complete_history_universe,
    entry_eligible_universe,
    prepare_model_panel,
)
from cross_sectional_ml.utils import read_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the final model panel.")
    parser.add_argument("--protocol", choices=("o2o", "de"), required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--eligibility", required=True, help="Entry-session eligibility panel.")
    parser.add_argument("--calendar", required=True, help="Trading calendar with a Date column.")
    parser.add_argument("--splits", required=True, help="Date-role table from build_splits.py.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = read_frame(args.features)
    labels = read_frame(args.labels)
    manifest = load_feature_manifest(args.protocol)
    label = get_label_spec(args.protocol)
    calendar = read_frame(args.calendar)
    trading_dates = calendar["Date"].tolist()
    roles = read_frame(args.splits)[["signal_date", "entry_date"]].copy()
    for column in roles.columns:
        roles[column] = normalize_date_series(roles[column])
    if roles.isna().any(axis=None):
        raise ValueError("date-role table contains invalid signal or entry dates")
    roles = roles.astype("int64").drop_duplicates()
    conflicts = roles.groupby("entry_date")["signal_date"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("an entry date maps to more than one signal date")
    entry_to_signal = roles.set_index("entry_date")["signal_date"].to_dict()

    entry_panel = read_frame(args.eligibility)
    if "Date" not in entry_panel.columns:
        raise ValueError("eligibility table is missing the Date column")
    entry_dates = normalize_date_series(entry_panel["Date"])
    entry_panel = entry_panel.loc[entry_dates.isin(entry_to_signal)].copy()
    eligible = entry_eligible_universe(entry_panel, entry_to_signal)
    history = complete_history_universe(
        features,
        trading_dates,
        sequence_length=SEQUENCE_LENGTH,
    )
    panel = prepare_model_panel(
        features,
        labels,
        manifest.columns,
        raw_return_column=label.raw_column,
        target_column=label.target_column,
        eligible_keys=eligible,
        history_keys=history,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    print(f"wrote {len(panel):,} final stock-days to {args.output}")


if __name__ == "__main__":
    main()
