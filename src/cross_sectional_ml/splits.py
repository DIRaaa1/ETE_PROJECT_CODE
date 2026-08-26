"""Fixed two-fold rolling design with trading-calendar endpoint mapping."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_years: tuple[int, int]
    validation_years: tuple[int, int]
    test_years: tuple[int, int]


@dataclass(frozen=True)
class DateRole:
    fold_id: str
    split: str
    signal_date: int
    entry_date: int
    exit_date: int
    purged: bool


F1 = Fold("F1", (2016, 2022), (2023, 2023), (2024, 2024))
F2 = Fold("F2", (2017, 2023), (2024, 2024), (2025, 2025))
FIXED_FOLDS = (F1, F2)
PURGE_SIGNAL_DATES = 3


def year(date: int) -> int:
    """Extract the calendar year from an integer YYYYMMDD date."""

    value = int(date)
    if value < 10_000_000 or value > 99_991_231:
        raise ValueError(f"date is not an eight-digit YYYYMMDD value: {date!r}")
    return value // 10_000


def dates_in_years(dates: Iterable[int], years: tuple[int, int]) -> list[int]:
    """Select unique sorted signal dates in an inclusive year range."""

    start, end = years
    if start > end:
        raise ValueError(f"invalid year range: {years}")
    return [date for date in sorted(set(map(int, dates))) if start <= year(date) <= end]


def map_endpoints(
    signal_dates: Sequence[int],
    trading_dates: Sequence[int],
) -> dict[int, tuple[int, int]]:
    """Map signal t to the next two sessions (entry t+1, exit t+2)."""

    calendar = sorted(set(map(int, trading_dates)))
    position = {date: index for index, date in enumerate(calendar)}
    mapped: dict[int, tuple[int, int]] = {}
    for signal_date in sorted(set(map(int, signal_dates))):
        index = position.get(signal_date)
        if index is not None and index + 2 < len(calendar):
            mapped[signal_date] = (calendar[index + 1], calendar[index + 2])
    return mapped


def build_date_roles(
    common_signal_dates: Sequence[int],
    trading_dates: Sequence[int],
    fold: Fold,
    purge_count: int = PURGE_SIGNAL_DATES,
) -> list[DateRole]:
    """Build train/validation/test rows and mark each partition's final dates."""

    if purge_count < 0:
        raise ValueError("purge_count cannot be negative")
    endpoints = map_endpoints(common_signal_dates, trading_dates)
    rows: list[DateRole] = []
    split_ranges = (
        ("train", fold.train_years),
        ("validation", fold.validation_years),
        ("test", fold.test_years),
    )
    for split, years in split_ranges:
        split_dates = dates_in_years(common_signal_dates, years)
        missing_endpoints = [date for date in split_dates if date not in endpoints]
        if missing_endpoints:
            raise ValueError(
                f"{fold.fold_id}/{split}: signal dates lack t+1/t+2 calendar endpoints: "
                f"{missing_endpoints[:10]}"
            )
        if len(split_dates) <= purge_count:
            raise ValueError(
                f"{fold.fold_id}/{split}: {len(split_dates)} dates cannot support "
                f"a {purge_count}-date purge and a non-empty active partition"
            )
        purged_dates = set(split_dates[-purge_count:]) if purge_count else set()
        for signal_date in split_dates:
            entry_date, exit_date = endpoints[signal_date]
            rows.append(
                DateRole(
                    fold_id=fold.fold_id,
                    split=split,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_date=exit_date,
                    purged=signal_date in purged_dates,
                )
            )
    return rows


def validate_date_roles(
    rows: Sequence[DateRole],
    purge_count: int = PURGE_SIGNAL_DATES,
) -> None:
    """Validate non-overlap, endpoint order, and exactly three tail purges."""

    if not rows:
        raise ValueError("date-role table is empty")
    for fold_id in sorted({row.fold_id for row in rows}):
        fold_rows = [row for row in rows if row.fold_id == fold_id]
        for row in fold_rows:
            if row.split not in {"train", "validation", "test"}:
                raise ValueError(f"{fold_id}: unknown split {row.split!r}")
            if not row.signal_date < row.entry_date < row.exit_date:
                raise ValueError(f"{fold_id}: endpoint order is not signal < entry < exit: {row}")
        active_sets = {
            split: {row.signal_date for row in fold_rows if row.split == split and not row.purged}
            for split in ("train", "validation", "test")
        }
        if any(not dates for dates in active_sets.values()):
            raise ValueError(f"{fold_id} contains an empty active partition")
        if active_sets["train"] & active_sets["validation"]:
            raise ValueError(f"{fold_id}: train/validation overlap")
        if active_sets["train"] & active_sets["test"]:
            raise ValueError(f"{fold_id}: train/test overlap")
        if active_sets["validation"] & active_sets["test"]:
            raise ValueError(f"{fold_id}: validation/test overlap")

        for split in ("train", "validation", "test"):
            split_rows = sorted(
                (row for row in fold_rows if row.split == split),
                key=lambda row: row.signal_date,
            )
            purged = [row for row in split_rows if row.purged]
            if len(purged) != purge_count:
                raise ValueError(f"{fold_id}/{split}: expected {purge_count} purged dates, got {len(purged)}")
            expected_purged = [row.signal_date for row in split_rows[-purge_count:]] if purge_count else []
            if [row.signal_date for row in purged] != expected_purged:
                raise ValueError(f"{fold_id}/{split}: purge is not exactly the partition tail")


def build_fixed_rolling_splits(
    common_signal_dates: Sequence[int],
    trading_dates: Sequence[int],
    *,
    purge_count: int = PURGE_SIGNAL_DATES,
) -> list[DateRole]:
    """Build and validate both registered folds in F1, F2 order."""

    rows = [
        row
        for fold in FIXED_FOLDS
        for row in build_date_roles(common_signal_dates, trading_dates, fold, purge_count)
    ]
    validate_date_roles(rows, purge_count)
    return rows


def date_roles_frame(rows: Sequence[DateRole], *, include_purged: bool = True) -> pd.DataFrame:
    """Convert immutable date roles to a convenient tabular representation."""

    selected = rows if include_purged else [row for row in rows if not row.purged]
    columns = ("fold_id", "split", "signal_date", "entry_date", "exit_date", "purged")
    return pd.DataFrame((asdict(row) for row in selected), columns=columns)
