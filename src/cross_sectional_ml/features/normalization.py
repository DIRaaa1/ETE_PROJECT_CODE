"""Frozen daily cross-sectional feature normalization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

MAD_NORMAL_SCALE = 1.4826
MAD_MULTIPLIER = 3.0


def legacy_mad3_v1(
    values: np.ndarray,
    *,
    min_valid_count: int = 10,
    column_block_size: int = 64,
) -> np.ndarray:
    """Reproduce the archived MAD3-clip-then-z-score transform.

    This historical version deliberately divides the clipped squared-deviation
    sum by ``n_rows - 1``, not by ``finite_count - 1``.  Missing inputs and
    degenerate columns become registered zeroes in the float32 output.
    """

    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"values must be two-dimensional, got shape {matrix.shape}")
    n_rows, n_columns = matrix.shape
    output = np.zeros((n_rows, n_columns), dtype="float32")
    if n_rows == 0 or n_columns == 0:
        return output
    if min_valid_count < 1:
        raise ValueError("min_valid_count must be positive")
    block_size = max(1, int(column_block_size))
    width_scale = np.float32(MAD_MULTIPLIER * MAD_NORMAL_SCALE)
    legacy_denominator = max(n_rows - 1, 1)

    for start in range(0, n_columns, block_size):
        end = min(start + block_size, n_columns)
        block = matrix[:, start:end].astype("float32", copy=True)
        block[~np.isfinite(block)] = np.nan
        initial_finite = np.isfinite(block)
        valid_count = initial_finite.sum(axis=0)
        enough = valid_count >= int(min_valid_count)
        if not enough.any():
            continue

        selected_columns = np.arange(start, end, dtype=np.int64)[enough]
        selected = block[:, enough]
        with np.errstate(all="ignore"):
            median = np.nanmedian(selected, axis=0).astype("float32", copy=False)
            mad = np.nanmedian(np.abs(selected - median), axis=0).astype("float32", copy=False)
        width = width_scale * mad
        valid_width = np.isfinite(median) & np.isfinite(width) & (width > 0)
        if not valid_width.any():
            continue

        selected = selected[:, valid_width]
        median = median[valid_width]
        width = width[valid_width]
        selected_columns = selected_columns[valid_width]
        np.clip(selected, median - width, median + width, out=selected)
        finite = np.isfinite(selected)
        count = finite.sum(axis=0)
        sums = np.where(finite, selected, 0.0).sum(axis=0, dtype=np.float64)
        mean = sums / np.maximum(count, 1)
        valid_mean = np.isfinite(mean) & (count > 0)
        if not valid_mean.any():
            continue

        selected = selected[:, valid_mean]
        finite = finite[:, valid_mean]
        mean = mean[valid_mean]
        selected_columns = selected_columns[valid_mean]
        centered = np.where(finite, selected - mean, 0.0)
        standard_deviation = np.sqrt((centered * centered).sum(axis=0, dtype=np.float64) / legacy_denominator)
        valid_standard_deviation = np.isfinite(standard_deviation) & (standard_deviation > 0)
        if not valid_standard_deviation.any():
            continue

        zscore = (
            selected[:, valid_standard_deviation] - mean[valid_standard_deviation]
        ) / standard_deviation[valid_standard_deviation]
        zscore[~finite[:, valid_standard_deviation]] = 0.0
        output[:, selected_columns[valid_standard_deviation]] = zscore.astype("float32", copy=False)
    return output


def mad3_zscore_matrix(
    values: np.ndarray,
    min_valid_count: int = 10,
    col_block_size: int = 64,
) -> np.ndarray:
    """Compatibility spelling for :func:`legacy_mad3_v1`."""

    return legacy_mad3_v1(
        values,
        min_valid_count=min_valid_count,
        column_block_size=col_block_size,
    )


def standardize_feature_frame_legacy_v1(
    frame: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    *,
    suffix: str = "mad3_zscore",
    min_valid_count: int = 10,
    column_block_size: int = 64,
) -> pd.DataFrame:
    """Normalize selected feature columns while preserving keys and row order."""

    if "Date" in frame.columns and frame["Date"].nunique(dropna=False) > 1:
        raise ValueError("feature normalization expects exactly one signal date")

    if feature_columns is None:
        columns = [
            column
            for column in frame.columns
            if column not in {"Code", "Date"}
            and column.startswith(("M30_", "D1_", "EO_"))
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
    else:
        columns = list(feature_columns)
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns contains duplicates")
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"feature columns are absent from frame: {missing}")

    normalized = legacy_mad3_v1(
        frame.loc[:, columns].to_numpy(dtype="float32", copy=True),
        min_valid_count=min_valid_count,
        column_block_size=column_block_size,
    )
    suffix = suffix.strip()
    output_columns = [column if not suffix else f"{column}_{suffix}" for column in columns]
    feature_set = set(columns)
    base_columns = [column for column in frame.columns if column not in feature_set]
    output = frame.loc[:, base_columns].reset_index(drop=True).copy()
    return pd.concat(
        [output, pd.DataFrame(normalized, columns=output_columns, index=output.index)],
        axis=1,
    )
