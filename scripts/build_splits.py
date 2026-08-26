from __future__ import annotations

import argparse
from pathlib import Path

from cross_sectional_ml.splits import build_fixed_rolling_splits, date_roles_frame
from cross_sectional_ml.utils import read_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed F1/F2 date-role table.")
    parser.add_argument("--signal-dates", required=True, help="Table containing a Date column.")
    parser.add_argument("--calendar", required=True, help="Trading calendar containing a Date column.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signal_dates = read_frame(args.signal_dates)["Date"].tolist()
    trading_dates = read_frame(args.calendar)["Date"].tolist()
    rows = build_fixed_rolling_splits(signal_dates, trading_dates)
    frame = date_roles_frame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {len(frame):,} date roles to {args.output}")


if __name__ == "__main__":
    main()
