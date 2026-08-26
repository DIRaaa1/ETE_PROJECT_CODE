from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .common import tree_metric_values
from .registry import merge_model_config

if TYPE_CHECKING:
    from cross_sectional_ml.training.data import DataPack


class XGBoostRegressor:
    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42) -> None:
        self.config = merge_model_config("xgb", config)
        self.seed = int(seed)
        self.booster: Any | None = None
        self.evals_result_: dict[str, Any] = {}
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None

    def fit(self, train: DataPack, valid: DataPack) -> XGBoostRegressor:
        import xgboost as xgb

        train_y = train.require_labels()
        valid_y = valid.require_labels()
        if tuple(train.feature_names) != tuple(valid.feature_names):
            raise ValueError("Training and validation feature schemas differ")
        params = {
            "objective": self.config["objective"],
            "eta": float(self.config["learning_rate"]),
            "max_depth": int(self.config["max_depth"]),
            "min_child_weight": float(self.config["min_child_weight"]),
            "subsample": float(self.config["subsample"]),
            "colsample_bytree": float(self.config["colsample_bytree"]),
            "lambda": float(self.config["lambda"]),
            "alpha": float(self.config["alpha"]),
            "tree_method": self.config["tree_method"],
            "max_bin": int(self.config["max_bin"]),
            "nthread": int(self.config["nthread"]),
            "seed": self.seed,
            "disable_default_eval_metric": 1,
        }
        feature_names = list(train.feature_names)
        train_matrix = xgb.QuantileDMatrix(
            train.x,
            label=train_y,
            feature_names=feature_names,
            max_bin=params["max_bin"],
        )
        valid_matrix = xgb.QuantileDMatrix(
            valid.x,
            label=valid_y,
            feature_names=feature_names,
            max_bin=params["max_bin"],
            ref=train_matrix,
        )

        def daily_metrics(predictions: np.ndarray, _matrix: Any) -> list[tuple[str, float]]:
            values = tree_metric_values(predictions, valid)
            return [
                ("DailyMSEMean", values["DailyMSEMean"]),
                ("DailyR2Mean", values["DailyR2Mean"]),
                ("DailyICMean", values["DailyICMean"]),
            ]

        self.evals_result_ = {}
        self.booster = xgb.train(
            params,
            train_matrix,
            num_boost_round=int(self.config["num_boost_round"]),
            evals=[(valid_matrix, "valid")],
            custom_metric=daily_metrics,
            maximize=True,
            early_stopping_rounds=int(self.config["early_stopping_rounds"]),
            evals_result=self.evals_result_,
            verbose_eval=self.config.get("verbose_eval", False),
        )
        self.best_iteration_ = int(self.booster.best_iteration)
        self.best_score_ = tree_metric_values(self.predict(valid), valid)["DailyICMean"]
        return self

    def predict(self, data: DataPack | np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("XGBoost model is not fitted")
        import xgboost as xgb

        x = data.x if hasattr(data, "x") else np.asarray(data)
        feature_names = list(data.feature_names) if hasattr(data, "feature_names") else None
        matrix = xgb.DMatrix(x, feature_names=feature_names)
        if self.best_iteration_ is not None:
            predictions = self.booster.predict(matrix, iteration_range=(0, self.best_iteration_ + 1))
        else:
            predictions = self.booster.predict(matrix)
        return np.asarray(predictions, dtype=np.float64).reshape(-1)

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.booster is None:
            raise RuntimeError("XGBoost model is not fitted")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(output))
        return output

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, config: dict[str, Any] | None = None, seed: int = 42
    ) -> XGBoostRegressor:
        import xgboost as xgb

        model = cls(config=config, seed=seed)
        model.booster = xgb.Booster()
        model.booster.load_model(str(path))
        best_iteration = model.booster.attr("best_iteration")
        model.best_iteration_ = int(best_iteration) if best_iteration is not None else None
        return model
