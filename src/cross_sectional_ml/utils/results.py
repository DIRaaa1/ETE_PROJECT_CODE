from __future__ import annotations

import numpy as np
import pandas as pd


def prediction_frame(
    dataset: object,
    predictions: np.ndarray,
    *,
    model: str,
    fold: str,
    target: np.ndarray | None = None,
    returns: np.ndarray | None = None,
) -> pd.DataFrame:
    values: dict[str, object] = {
        "model": model,
        "fold": fold,
        "Date": np.asarray(dataset.dates),
        "Code": np.asarray(dataset.symbols).astype(str),
        "prediction": predictions,
    }
    if target is not None:
        values["target"] = target
    if returns is not None:
        values["raw_return"] = returns
    return pd.DataFrame(values)
