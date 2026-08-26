from __future__ import annotations

import argparse
from pathlib import Path

from cross_sectional_ml.labels import build_vwap_label
from cross_sectional_ml.utils import read_minute_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one O2O or DE return-label panel.")
    parser.add_argument("--protocol", choices=("o2o", "de"), required=True)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--entry", required=True, help="Entry-session file or minute directory.")
    parser.add_argument("--exit", required=True, help="Exit-session file or minute directory.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = build_vwap_label(
        read_minute_session(args.entry),
        read_minute_session(args.exit),
        protocol=args.protocol,
        signal_date=args.signal_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    label.to_parquet(args.output, index=False)
    print(f"wrote {len(label):,} labels to {args.output}")


if __name__ == "__main__":
    main()
