from __future__ import annotations

from collections import defaultdict

from cross_sectional_ml.splits import (
    FIXED_FOLDS,
    PURGE_SIGNAL_DATES,
    build_fixed_rolling_splits,
    date_roles_frame,
    map_endpoints,
    validate_date_roles,
    year,
)


def _calendar() -> tuple[list[int], list[int]]:
    signal_dates = [
        year_value * 10_000 + 100 + day for year_value in range(2016, 2026) for day in range(2, 8)
    ]
    # Two following sessions are needed to map the final 2025 signals.
    trading_dates = [*signal_dates, 20260102, 20260103]
    return signal_dates, trading_dates


def test_registered_fold_years_are_fixed() -> None:
    assert [fold.fold_id for fold in FIXED_FOLDS] == ["F1", "F2"]
    assert FIXED_FOLDS[0].train_years == (2016, 2022)
    assert FIXED_FOLDS[0].validation_years == (2023, 2023)
    assert FIXED_FOLDS[0].test_years == (2024, 2024)
    assert FIXED_FOLDS[1].train_years == (2017, 2023)
    assert FIXED_FOLDS[1].validation_years == (2024, 2024)
    assert FIXED_FOLDS[1].test_years == (2025, 2025)


def test_each_partition_purges_exactly_its_final_three_signal_dates() -> None:
    signal_dates, trading_dates = _calendar()
    rows = build_fixed_rolling_splits(signal_dates, trading_dates)
    validate_date_roles(rows)

    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for row in rows:
        grouped[(row.fold_id, row.split)].append(row)
    for partition_rows in grouped.values():
        partition_rows.sort(key=lambda row: row.signal_date)
        assert sum(row.purged for row in partition_rows) == PURGE_SIGNAL_DATES
        assert [row.signal_date for row in partition_rows if row.purged] == [
            row.signal_date for row in partition_rows[-PURGE_SIGNAL_DATES:]
        ]
        assert all(not row.purged for row in partition_rows[:-PURGE_SIGNAL_DATES])

    frame = date_roles_frame(rows, include_purged=False)
    assert not frame["purged"].any()
    assert set(frame["fold_id"]) == {"F1", "F2"}
    assert set(frame["split"]) == {"train", "validation", "test"}


def test_endpoint_mapping_uses_next_two_independent_calendar_sessions() -> None:
    calendar = [20240102, 20240103, 20240105, 20240108]
    mapped = map_endpoints([20240102, 20240103], calendar)

    assert mapped[20240102] == (20240103, 20240105)
    assert mapped[20240103] == (20240105, 20240108)
    assert year(20240108) == 2024
