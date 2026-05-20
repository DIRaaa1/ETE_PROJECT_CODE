import os
import sys
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List, Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.MonitorMetric import (
    build_monitor_frame,
    compute_monitor_metrics,
    get_metric_mode,
    metric_value_is_better,
    sort_model_records,
    save_monitor_history
)


class LightGBMModel:
    def __init__(
        self,
        feature_name: List[str],
        hyper_params: dict,
        time_column: str,
        symbol_column: str,
        seed: int = 42
    ):
        self.feature_name = list(feature_name)
        self.hyper_params = dict(hyper_params)
        self.model_name = "LightGBM"
        self.time_column = time_column
        self.symbol_column = symbol_column
        self.seed = int(seed)
        self.booster = None
        self.best_model_paths = []
        self.best_iteration = None
        self.best_monitor_value = None
        self.eval_history = []
        self.load_hyper_param(self.hyper_params)
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)

    def load_hyper_param(self, hyper_params: dict):
        self.seed = int(hyper_params.get("Seed", self.seed))

        self.objective = str(hyper_params.get("Objective", "regression"))
        self.metric = str(hyper_params.get("Metric", "None"))
        self.boosting_type = str(hyper_params.get("BoostingType", "gbdt"))

        self.device_type = str(hyper_params.get("DeviceType", "cpu"))
        self.num_threads = int(hyper_params.get("NumThreads", 16))

        self.learning_rate = float(hyper_params.get("LearningRate", 0.01))
        self.num_boost_round = int(hyper_params.get("NumBoostRound", 10000))
        self.early_stopping_rounds = int(hyper_params.get("EarlyStoppingRounds", 512))

        self.num_leaves = int(hyper_params.get("NumLeaves", 63))
        self.max_depth = int(hyper_params.get("MaxDepth", -1))
        self.min_data_in_leaf = int(hyper_params.get("MinDataInLeaf", 512))
        self.min_sum_hessian_in_leaf = float(hyper_params.get("MinSumHessianInLeaf", 0.001))

        self.feature_fraction = float(hyper_params.get("FeatureFraction", 0.8))
        self.feature_fraction_bynode = float(hyper_params.get("FeatureFractionBynode", 1.0))
        self.bagging_fraction = float(hyper_params.get("BaggingFraction", 0.8))
        self.bagging_freq = int(hyper_params.get("BaggingFreq", 1))

        self.lambda_l1 = float(hyper_params.get("LambdaL1", 0.0))
        self.lambda_l2 = float(hyper_params.get("LambdaL2", 1.0))
        self.min_gain_to_split = float(hyper_params.get("MinGainToSplit", 0.0))
        self.path_smooth = float(hyper_params.get("PathSmooth", 0.0))
        self.extra_trees = bool(hyper_params.get("ExtraTrees", False))

        self.max_bin = int(hyper_params.get("MaxBin", 255))
        self.min_data_in_bin = int(hyper_params.get("MinDataInBin", 3))

        self.deterministic = bool(hyper_params.get("Deterministic", True))
        self.force_col_wise = bool(hyper_params.get("ForceColWise", True))
        self.force_row_wise = bool(hyper_params.get("ForceRowWise", False))

        self.gpu_platform_id = int(hyper_params.get("GpuPlatformId", -1))
        self.gpu_device_id = int(hyper_params.get("GpuDeviceId", -1))
        self.gpu_use_dp = bool(hyper_params.get("GpuUseDp", False))

        self.monitor_metric_name = str(hyper_params.get("MonitorMetricName", "DailyMSEMean"))
        self.monitor_metric_mode = hyper_params.get("MonitorMetricMode", "auto")
        self.monitor_return_column = hyper_params.get("MonitorReturnColumn", None)
        self.monitor_rank_by = hyper_params.get("MonitorRankBy", "pred")
        self.monitor_rank_abs = bool(hyper_params.get("MonitorRankAbs", False))

        self.best_model_keep_num = int(hyper_params.get("BestModelKeepNum", 1))
        self.valid_metric_eval_name = str(hyper_params.get("ValidMetricEvalName", "monitor"))
        self.log_eval_period = int(hyper_params.get("LogEvalPeriod", 1))
        self.free_raw_data = bool(hyper_params.get("FreeRawData", False))

        if self.num_boost_round <= 0:
            raise ValueError("NumBoostRound Invalid >= 1")
        if self.early_stopping_rounds <= 0:
            raise ValueError("EarlyStoppingRounds Invalid >= 1")
        if self.num_leaves <= 1:
            raise ValueError("NumLeaves Invalid > 1")
        if self.min_data_in_leaf <= 0:
            raise ValueError("MinDataInLeaf Invalid >= 1")
        if self.learning_rate <= 0:
            raise ValueError("LearningRate Invalid > 0")

    def _metric_mode_higher_is_better(self, monitor_mode: str) -> bool:
        mode = str(monitor_mode).lower()
        if mode in {"max", "maximize", "higher", "larger", "greater", "up", "asc", "ascending"}:
            return True
        if mode in {"min", "minimize", "lower", "smaller", "less", "down", "desc", "descending"}:
            return False

        name = str(self.monitor_metric_name).lower()
        if any(k in name for k in ["loss", "mse", "rmse", "mae", "error"]):
            return False
        return True

    def _safe_metric_name(self, name: str) -> str:
        return str(name).replace("/", "_").replace(" ", "_").replace(":", "_")

    def _to_numpy_x(self, x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        return np.ascontiguousarray(x, dtype=np.float32)

    def _to_numpy_y(self, y: Any) -> np.ndarray:
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        y = np.asarray(y).reshape(-1)
        return np.ascontiguousarray(y, dtype=np.float32)

    def _build_lgb_params(self) -> Dict[str, Any]:
        params = {
            "objective": self.objective,
            "metric": self.metric,
            "boosting_type": self.boosting_type,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "min_sum_hessian_in_leaf": self.min_sum_hessian_in_leaf,
            "feature_fraction": self.feature_fraction,
            "feature_fraction_bynode": self.feature_fraction_bynode,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "min_gain_to_split": self.min_gain_to_split,
            "path_smooth": self.path_smooth,
            "extra_trees": self.extra_trees,
            "max_bin": self.max_bin,
            "min_data_in_bin": self.min_data_in_bin,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "data_random_seed": self.seed,
            "drop_seed": self.seed,
            "deterministic": self.deterministic,
            "force_col_wise": self.force_col_wise,
            "force_row_wise": self.force_row_wise,
            "num_threads": self.num_threads,
            "device_type": self.device_type,
            "verbosity": -1
        }

        if self.device_type.lower() == "gpu":
            params["gpu_platform_id"] = self.gpu_platform_id
            params["gpu_device_id"] = self.gpu_device_id
            params["gpu_use_dp"] = self.gpu_use_dp

        return params

    def _extract_meta(self, data: Any):
        dates = data.build_row_dates() if hasattr(data, "build_row_dates") else getattr(data, "dates", None)
        symbols = getattr(data, "symbols", None)
        monitor_return = getattr(data, "monitor_return", None)

        if dates is None:
            raise ValueError("valid_data Invalid dates / build_row_dates")
        if symbols is None:
            raise ValueError("valid_data Invalid symbols")

        if monitor_return is None:
            monitor_return = getattr(data, "y", None)

        if hasattr(monitor_return, "detach"):
            monitor_return = monitor_return.detach().cpu().numpy()

        dates = np.asarray(dates)
        symbols = np.asarray(symbols)
        monitor_return = np.asarray(monitor_return).reshape(-1)

        return dates, symbols, monitor_return

    def _make_valid_feval(
        self,
        valid_dates: np.ndarray,
        valid_symbols: np.ndarray,
        valid_monitor_return: np.ndarray,
        monitor_mode: str
    ):
        higher_is_better = self._metric_mode_higher_is_better(monitor_mode)
        monitor_eval_name = self._safe_metric_name(self.monitor_metric_name)

        def feval(preds, dataset):
            labels = dataset.get_label()
            preds = np.asarray(preds).reshape(-1)
            labels = np.asarray(labels).reshape(-1)

            monitor_df = build_monitor_frame(
                date_series=valid_dates,
                symbol_series=valid_symbols,
                pred_array=preds,
                label_array=labels,
                return_array=valid_monitor_return,
                date_col=self.time_column,
                symbol_col=self.symbol_column
            )

            metrics = compute_monitor_metrics(
                monitor_df,
                date_col=self.time_column,
                pred_col="y_pred",
                label_col="y_true",
                return_col="monitor_return",
                rank_by=self.monitor_rank_by,
                rank_abs=self.monitor_rank_abs
            )

            monitor_value_raw = float(metrics[self.monitor_metric_name])
            if np.isfinite(monitor_value_raw):
                monitor_value = monitor_value_raw
            else:
                monitor_value = -1e30 if higher_is_better else 1e30

            l2_value = float(np.mean((preds - labels) ** 2))

            row = {
                "iteration": len(self.eval_history) + 1,
                "monitor_metric_name": self.monitor_metric_name,
                "monitor_metric_mode": monitor_mode,
                "valid_monitor_value": monitor_value_raw if np.isfinite(monitor_value_raw) else np.nan,
                "valid_l2": l2_value
            }

            for k, v in metrics.items():
                row[f"valid_{k}"] = v

            self.eval_history.append(row)

            return [
                (monitor_eval_name, monitor_value, higher_is_better),
                ("L2", l2_value, False)
            ]

        return feval

    def train(
        self,
        train_data: Any,
        valid_data: Any,
        best_model_path: str = "best_lightgbm_model.txt"
    ):
        if train_data.num_rows == 0:
            raise ValueError("train_data Invalid ")
        if valid_data.num_rows == 0:
            raise ValueError("valid_data Invalid ")

        X_train = self._to_numpy_x(train_data.x)
        y_train = self._to_numpy_y(train_data.y)
        X_valid = self._to_numpy_x(valid_data.x)
        y_valid = self._to_numpy_y(valid_data.y)

        valid_dates, valid_symbols, valid_monitor_return = self._extract_meta(valid_data)

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train Invalid y_train Invalid: {X_train.shape[0]} vs {y_train.shape[0]}")
        if X_valid.shape[0] != y_valid.shape[0]:
            raise ValueError(f"X_valid Invalid y_valid Invalid: {X_valid.shape[0]} vs {y_valid.shape[0]}")
        if len(valid_dates) != X_valid.shape[0]:
            raise ValueError("valid_dates Invalid X_valid Invalid")
        if len(valid_symbols) != X_valid.shape[0]:
            raise ValueError("valid_symbols Invalid X_valid Invalid")
        if len(valid_monitor_return) != X_valid.shape[0]:
            raise ValueError("valid_monitor_return Invalid X_valid Invalid")

        params = self._build_lgb_params()

        lgb_train = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_name,
            free_raw_data=self.free_raw_data
        )

        lgb_valid = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=lgb_train,
            feature_name=self.feature_name,
            free_raw_data=self.free_raw_data
        )

        monitor_mode = get_metric_mode(self.monitor_metric_name, self.monitor_metric_mode)
        feval = self._make_valid_feval(
            valid_dates=valid_dates,
            valid_symbols=valid_symbols,
            valid_monitor_return=valid_monitor_return,
            monitor_mode=monitor_mode
        )

        evals_result = {}
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds,
                first_metric_only=True,
                verbose=True
            ),
            lgb.record_evaluation(evals_result)
        ]

        if self.log_eval_period > 0:
            callbacks.append(lgb.log_evaluation(period=self.log_eval_period))

        self.eval_history = []

        self.booster = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=[lgb_valid],
            valid_names=["valid"],
            feval=feval,
            callbacks=callbacks
        )

        monitor_eval_name = self._safe_metric_name(self.monitor_metric_name)

        if self.booster.best_iteration is not None and self.booster.best_iteration > 0:
            self.best_iteration = int(self.booster.best_iteration)
        else:
            self.best_iteration = int(self.booster.current_iteration())

        if len(self.eval_history) >= self.best_iteration:
            self.best_monitor_value = float(self.eval_history[self.best_iteration - 1]["valid_monitor_value"])
        else:
            valid_pred = self.booster.predict(X_valid, num_iteration=self.best_iteration)
            monitor_df = build_monitor_frame(
                date_series=valid_dates,
                symbol_series=valid_symbols,
                pred_array=valid_pred,
                label_array=y_valid,
                return_array=valid_monitor_return,
                date_col=self.time_column,
                symbol_col=self.symbol_column
            )
            valid_metrics = compute_monitor_metrics(
                monitor_df,
                date_col=self.time_column,
                pred_col="y_pred",
                label_col="y_true",
                return_col="monitor_return",
                rank_by=self.monitor_rank_by,
                rank_abs=self.monitor_rank_abs
            )
            self.best_monitor_value = float(valid_metrics[self.monitor_metric_name])

        safe_metric_name = self._safe_metric_name(self.monitor_metric_name)
        if best_model_path.endswith(".txt"):
            model_path = best_model_path.replace(
                ".txt",
                f"_iter{self.best_iteration}_monitor_{safe_metric_name}_{self.best_monitor_value:.8f}.txt"
            )
        else:
            model_path = f"{best_model_path}_iter{self.best_iteration}_monitor_{safe_metric_name}_{self.best_monitor_value:.8f}.txt"

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.booster.save_model(model_path, num_iteration=self.best_iteration)

        self.best_model_paths = [(self.best_monitor_value, model_path, self.best_iteration)]
        self.best_model_paths = sort_model_records(self.best_model_paths, monitor_mode)

        history_csv_path = best_model_path.replace("_best_model.txt", "_iteration_monitor_metrics.csv")
        if self.eval_history:
            history_rows = []
            for row in self.eval_history:
                out = dict(row)
                out["seed"] = self.seed
                out["best_iteration"] = self.best_iteration
                out["objective"] = self.objective
                out["boosting_type"] = self.boosting_type
                out["device_type"] = self.device_type
                out["learning_rate"] = self.learning_rate
                out["num_leaves"] = self.num_leaves
                out["max_depth"] = self.max_depth
                out["min_data_in_leaf"] = self.min_data_in_leaf
                out["feature_fraction"] = self.feature_fraction
                out["bagging_fraction"] = self.bagging_fraction
                out["bagging_freq"] = self.bagging_freq
                out["lambda_l1"] = self.lambda_l1
                out["lambda_l2"] = self.lambda_l2
                out["max_bin"] = self.max_bin
                history_rows.append(out)
            save_monitor_history(history_rows, history_csv_path)

        print(f"Best Iteration: {self.best_iteration}")
        print(f"Best Valid {self.monitor_metric_name}: {self.best_monitor_value:.8f}")
        print(f"Best Model Path: {model_path}")

    def test(self, x_data: Any) -> np.ndarray:
        X = self._to_numpy_x(x_data)

        if self.booster is None:
            if not hasattr(self, "best_model_paths") or len(self.best_model_paths) == 0:
                raise RuntimeError("No trained LightGBM model available for testing.")
            _, model_path, best_iteration = self.best_model_paths[0]
            booster = lgb.Booster(model_file=model_path)
            pred = booster.predict(X, num_iteration=int(best_iteration))
            return np.asarray(pred).reshape(-1, 1)

        pred = self.booster.predict(X, num_iteration=int(self.best_iteration))
        return np.asarray(pred).reshape(-1, 1)

    def save_model(self, model_dir: str):
        if self.booster is None:
            raise RuntimeError("Invalid LightGBM booster ")
        os.makedirs(os.path.dirname(model_dir), exist_ok=True)
        self.booster.save_model(model_dir, num_iteration=self.best_iteration)