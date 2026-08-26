from __future__ import annotations

import argparse
from pathlib import Path

from cross_sectional_ml.features import (
    build_raw_feature_panel,
    project_protocol_features,
    standardize_feature_frame_legacy_v1,
)
from cross_sectional_ml.utils import read_dated_directory, read_frame, read_minute_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one protocol feature panel.")
    parser.add_argument("--protocol", choices=("o2o", "de"), required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--daily", required=True, help="Daily-history Parquet file or directory.")
    parser.add_argument("--minutes", required=True, help="Signal-session file or minute directory.")
    parser.add_argument("--early-open", help="Next-session file or minute directory; required for DE.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily_path = Path(args.daily)
    daily = read_dated_directory(daily_path) if daily_path.is_dir() else read_frame(daily_path)
    minutes = read_minute_session(args.minutes)
    early_open = read_minute_session(args.early_open) if args.early_open else None
    raw = build_raw_feature_panel(
        daily,
        minutes,
        signal_date=args.signal_date,
        protocol=args.protocol,
        early_open_minutes=early_open,
    )
    standardized = standardize_feature_frame_legacy_v1(raw)
    selected = project_protocol_features(standardized, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    print(f"wrote {len(selected):,} rows and {selected.shape[1] - 2} features to {args.output}")


if __name__ == "__main__":
    main()
