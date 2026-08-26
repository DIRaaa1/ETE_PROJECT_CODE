from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .common import tree_metric_values
from .registry import merge_model_config

if TYPE_CHECKING:
    from cross_sectional_ml.training.data import DataPack


class LightGBMRegressor:
    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42) -> None:
        self.config = merge_model_config("lgb", config)
        self.seed = int(seed)
        self.booster: Any | None = None
        self.evals_result_: dict[str, Any] = {}
        self.best_iteration_: int | None = None
        self.best_score_: float | None = None

    def fit(self, train: DataPack, valid: DataPack) -> LightGBMRegressor:
        import lightgbm as lgb

        train_y = train.require_labels()
        valid.require_labels()
        if tuple(train.feature_names) != tuple(valid.feature_names):
            raise ValueError("Training and validation feature schemas differ")
        params = {
            "objective": self.config["objective"],
            "metric": "None",
            "learning_rate": float(self.config["learning_rate"]),
            "num_leaves": int(self.config["num_leaves"]),
            "max_depth": int(self.config["max_depth"]),
            "min_data_in_leaf": int(self.config["min_data_in_leaf"]),
            "feature_fraction": float(self.config["feature_fraction"]),
            "bagging_fraction": float(self.config["bagging_fraction"]),
            "bagging_freq": int(self.config["bagging_freq"]),
            "lambda_l1": float(self.config["lambda_l1"]),
            "lambda_l2": float(self.config["lambda_l2"]),
            "num_threads": int(self.config["num_threads"]),
            "verbosity": -1,
            "seed": self.seed,
            "feature_pre_filter": False,
            "force_col_wise": True,
        }
        feature_names = list(train.feature_names)
        train_set = lgb.Dataset(train.x, label=train_y, feature_name=feature_names)
        valid_set = lgb.Dataset(
            valid.x,
            label=valid.require_labels(),
            reference=train_set,
            feature_name=feature_names,
        )

        def daily_metrics(predictions: np.ndarray, _dataset: Any) -> list[tuple[str, float, bool]]:
            values = tree_metric_values(predictions, valid)
            return [
                ("DailyICMean", values["DailyICMean"], True),
                ("DailyMSEMean", values["DailyMSEMean"], False),
                ("DailyR2Mean", values["DailyR2Mean"], True),
            ]

        self.evals_result_ = {}
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=int(self.config["num_boost_round"]),
            valid_sets=[valid_set],
            valid_names=["valid"],
            feval=daily_metrics,
            callbacks=[
                lgb.record_evaluation(self.evals_result_),
                lgb.early_stopping(
                    int(self.config["early_stopping_rounds"]),
                    first_metric_only=True,
                    verbose=bool(self.config.get("verbose", False)),
                ),
                lgb.log_evaluation(period=int(self.config.get("log_evaluation", 0))),
            ],
        )
        self.best_iteration_ = int(self.booster.best_iteration or self.booster.current_iteration())
        self.best_score_ = tree_metric_values(self.predict(valid), valid)["DailyICMean"]
        return self

    def predict(self, data: DataPack | np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("LightGBM model is not fitted")
        x = data.x if hasattr(data, "x") else np.asarray(data)
        return np.asarray(
            self.booster.predict(x, num_iteration=self.best_iteration_), dtype=np.float64
        ).reshape(-1)

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.booster is None:
            raise RuntimeError("LightGBM model is not fitted")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(output), num_iteration=self.best_iteration_)
        return output

    @classmethod
    def load_checkpoint(
        cls, path: str | Path, config: dict[str, Any] | None = None, seed: int = 42
    ) -> LightGBMRegressor:
        import lightgbm as lgb

        model = cls(config=config, seed=seed)
        model.booster = lgb.Booster(model_file=str(path))
        model.best_iteration_ = model.booster.current_iteration()
        return model
