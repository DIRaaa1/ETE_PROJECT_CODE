from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _paired_finite(predictions: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have equal length")
    finite = np.isfinite(predictions) & np.isfinite(labels)
    return predictions[finite], labels[finite]


def safe_corr(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions, labels = _paired_finite(predictions, labels)
    if len(predictions) < 2:
        return 0.0
    if np.unique(predictions).size == 1 or np.unique(labels).size == 1:
        return 0.0

    def unit_center(values: np.ndarray) -> np.ndarray | None:
        scale = float(np.max(np.abs(values)))
        scaled = values / scale if scale > 0.0 else values
        centered = scaled - scaled.mean()
        norm = float(np.linalg.norm(centered))
        return None if not np.isfinite(norm) or norm <= 0.0 else centered / norm

    prediction_unit = unit_center(predictions)
    label_unit = unit_center(labels)
    if prediction_unit is None or label_unit is None:
        return float("nan")
    return float(np.clip(np.dot(prediction_unit, label_unit), -1.0, 1.0))


def safe_mse(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions, labels = _paired_finite(predictions, labels)
    if len(predictions) == 0:
        return 0.0
    return float(np.mean((predictions - labels) ** 2))


def safe_r2(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions, labels = _paired_finite(predictions, labels)
    if len(predictions) < 2:
        return 0.0
    residual_sum = float(np.sum((labels - predictions) ** 2))
    total_sum = float(np.sum((labels - labels.mean()) ** 2))
    if total_sum <= 0.0 or not np.isfinite(total_sum):
        return 0.0
    return float(1.0 - residual_sum / total_sum)


def _daily_values(
    predictions: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    metric,
) -> np.ndarray:
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    dates = np.asarray(dates).reshape(-1)
    if not (len(predictions) == len(labels) == len(dates)):
        raise ValueError("predictions, labels, and dates must have equal length")
    return np.asarray(
        [metric(predictions[dates == date], labels[dates == date]) for date in np.unique(dates)],
        dtype=np.float64,
    )


def _finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def daily_ic_mean(predictions: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> float:
    return _finite_mean(_daily_values(predictions, labels, dates, safe_corr))


def daily_mse_mean(predictions: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> float:
    return _finite_mean(_daily_values(predictions, labels, dates, safe_mse))


def daily_r2_mean(predictions: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> float:
    return _finite_mean(_daily_values(predictions, labels, dates, safe_r2))


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    daily_ic_mean: float
    daily_mse_mean: float
    daily_r2_mean: float
    num_dates: int
    num_rows: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "DailyICMean": self.daily_ic_mean,
            "DailyMSEMean": self.daily_mse_mean,
            "DailyR2Mean": self.daily_r2_mean,
            "NumDates": self.num_dates,
            "NumRows": self.num_rows,
        }


def regression_metrics(predictions: np.ndarray, labels: np.ndarray, dates: np.ndarray) -> RegressionMetrics:
    dates = np.asarray(dates).reshape(-1)
    return RegressionMetrics(
        daily_ic_mean=daily_ic_mean(predictions, labels, dates),
        daily_mse_mean=daily_mse_mean(predictions, labels, dates),
        daily_r2_mean=daily_r2_mean(predictions, labels, dates),
        num_dates=len(np.unique(dates)),
        num_rows=len(dates),
    )


def compute_regression_metrics(
    predictions: np.ndarray, labels: np.ndarray, dates: np.ndarray
) -> dict[str, float | int]:
    return regression_metrics(predictions, labels, dates).as_dict()
