"""Prediction input and point-in-time panel alignment.

The functions in this module deliberately stop at data contracts.  They do not
drop names because an outcome is unavailable: outcomes are attached with a
left join only after the eight prediction panels have been aligned.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

from cross_sectional_ml.io import normalize_code_series, normalize_datetime_series
from cross_sectional_ml.models import MODEL_NAMES
from cross_sectional_ml.utils import read_frame

EIGHT_MODELS = MODEL_NAMES

FrameSource = pd.DataFrame | str | PathLike[str]


def read_model_predictions(root: str | Path, model: str, split: str = "test") -> pd.DataFrame:
    base = Path(root)
    direct = base / f"{model}.parquet"
    if direct.is_file():
        return read_frame(direct)
    parts = [base / fold / model / f"{split}_predictions.parquet" for fold in ("F1", "F2")]
    return pd.concat([read_frame(path) for path in parts], ignore_index=True)


def _read_table(source: FrameSource, *, code_column: str) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    return read_frame(source, code_column=code_column)


def _normalise_keys(
    frame: pd.DataFrame,
    *,
    date_column: str,
    code_column: str,
) -> pd.DataFrame:
    missing = {date_column, code_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"missing key columns: {sorted(missing)}")

    out = frame.copy()
    out[date_column] = normalize_datetime_series(out[date_column])
    out[code_column] = normalize_code_series(out[code_column])
    if out[date_column].isna().any():
        raise ValueError(f"{date_column} contains missing values")
    if out[code_column].isna().any() or out[code_column].eq("").any():
        raise ValueError(f"{code_column} contains missing or empty values")
    if out.duplicated([date_column, code_column]).any():
        duplicate = out.loc[
            out.duplicated([date_column, code_column], keep=False),
            [date_column, code_column],
        ].head(3)
        raise ValueError(f"duplicate Date/Code prediction keys: {duplicate.to_dict('records')}")
    return out.sort_values([date_column, code_column], kind="mergesort").reset_index(drop=True)


def read_prediction(
    source: FrameSource,
    *,
    prediction_column: str = "prediction",
    output_column: str = "score",
    date_column: str = "Date",
    code_column: str = "Code",
) -> pd.DataFrame:
    """Read one model's predictions and return sorted ``Date, Code, score`` rows.

    Prediction scores must be finite and each Date/Code key must be unique.
    CSV security codes are read as strings so leading zeroes are retained.
    """

    if output_column in {date_column, code_column}:
        raise ValueError("output_column must not replace a key column")
    frame = _read_table(source, code_column=code_column)
    if prediction_column not in frame.columns:
        raise ValueError(f"missing prediction column: {prediction_column!r}")
    frame = _normalise_keys(
        frame[[date_column, code_column, prediction_column]],
        date_column=date_column,
        code_column=code_column,
    )
    scores = pd.to_numeric(frame[prediction_column], errors="raise").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    if not np.isfinite(scores).all():
        raise ValueError("prediction scores must all be finite")
    return frame[[date_column, code_column]].assign(**{output_column: scores})


def align_eight_model_predictions(
    sources: Mapping[str, FrameSource],
    *,
    prediction_columns: str | Mapping[str, str] = "prediction",
    date_column: str = "Date",
    code_column: str = "Code",
) -> pd.DataFrame:
    """Read and exactly align the dissertation's eight model panels.

    No inner intersection is permitted.  Every model must have the same full
    ex-ante Date/Code universe; the returned model columns follow
    :data:`EIGHT_MODELS` regardless of mapping insertion order.
    """

    supplied = set(sources)
    expected = set(EIGHT_MODELS)
    if supplied != expected:
        raise ValueError(
            "sources must contain exactly the eight dissertation models; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )

    aligned: pd.DataFrame | None = None
    reference_keys: pd.MultiIndex | None = None
    for model in EIGHT_MODELS:
        prediction_column = (
            prediction_columns[model] if isinstance(prediction_columns, Mapping) else prediction_columns
        )
        one = read_prediction(
            sources[model],
            prediction_column=prediction_column,
            output_column=model,
            date_column=date_column,
            code_column=code_column,
        )
        keys = pd.MultiIndex.from_frame(one[[date_column, code_column]])
        if aligned is None:
            aligned = one
            reference_keys = keys
            continue
        assert reference_keys is not None
        if not keys.equals(reference_keys):
            missing = reference_keys.difference(keys)
            extra = keys.difference(reference_keys)
            raise ValueError(
                f"{model} Date/Code keys do not match the reference panel "
                f"(missing={len(missing)}, extra={len(extra)})"
            )
        aligned[model] = one[model].to_numpy(dtype=float)

    assert aligned is not None
    return aligned[[date_column, code_column, *EIGHT_MODELS]]


def attach_outcomes(
    predictions: pd.DataFrame,
    outcomes: FrameSource,
    *,
    outcome_column: str = "return",
    date_column: str = "Date",
    code_column: str = "Code",
) -> pd.DataFrame:
    """Left-align realised returns without changing the prediction universe.

    Missing and non-finite outcomes become ``NaN``.  They remain in the panel
    so portfolio accounting can retain their deployed notional and assign zero
    PnL without selecting a replacement.
    """

    if outcome_column in predictions.columns:
        raise ValueError(f"predictions already contain {outcome_column!r}")
    left = _normalise_keys(
        predictions,
        date_column=date_column,
        code_column=code_column,
    )
    raw_outcomes = _read_table(outcomes, code_column=code_column)
    if outcome_column not in raw_outcomes.columns:
        raise ValueError(f"missing outcome column: {outcome_column!r}")
    right = _normalise_keys(
        raw_outcomes[[date_column, code_column, outcome_column]],
        date_column=date_column,
        code_column=code_column,
    )
    realised = pd.to_numeric(right[outcome_column], errors="raise").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    realised[~np.isfinite(realised)] = np.nan
    right[outcome_column] = realised

    merged = left.merge(
        right,
        on=[date_column, code_column],
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(merged) != len(left):
        raise AssertionError("left outcome alignment changed the prediction universe")
    return merged
