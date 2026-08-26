from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cross_sectional_ml.features import load_feature_manifest
from cross_sectional_ml.io import normalize_panel_keys
from cross_sectional_ml.labels import get_label_spec
from cross_sectional_ml.models import (
    MODEL_NAMES,
    SEQUENCE_LENGTH,
    SEQUENCE_TORCH_MODELS,
    TREE_MODELS,
    build_model,
)
from cross_sectional_ml.training import LazySequenceDataset, train_torch_model
from cross_sectional_ml.universe import data_pack_from_panel
from cross_sectional_ml.utils import (
    ProjectPaths,
    prediction_frame,
    read_active_split_dates,
    read_frame,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one registered model and rolling fold.")
    parser.add_argument("--paths", type=Path, help="Optional project path configuration.")
    parser.add_argument("--panel", type=Path, help="Prepared 2016-2025 model panel.")
    parser.add_argument("--splits", type=Path, help="Date-role table from build_splits.py.")
    parser.add_argument("--protocol", choices=("o2o", "de"), required=True)
    parser.add_argument("--fold", choices=("F1", "F2"), required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--output", type=Path, help="Training output directory.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, help="Optional smoke-run override.")
    parser.add_argument("--batch-size", type=int, help="Optional runtime override.")
    parser.add_argument("--threads", type=int, help="Optional tree-thread override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.load(args.paths) if args.paths else None
    panel_path = args.panel or (paths.panel(args.protocol) if paths else None)
    splits_path = args.splits or (paths.splits if paths else None)
    output = args.output or (paths.train_dir(args.protocol, args.fold, args.model) if paths else None)
    if panel_path is None or splits_path is None or output is None:
        raise ValueError("provide --panel, --splits, and --output, or derive them with --paths")

    panel = normalize_panel_keys(read_frame(panel_path))
    manifest = load_feature_manifest(args.protocol)
    label = get_label_spec(args.protocol)
    required = {
        *manifest.columns,
        label.target_column,
        label.raw_column,
        "is_eligible",
        "is_scoreable",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"prepared panel is missing {len(missing)} columns: {sorted(missing)[:10]}")

    all_dates = sorted(panel["Date"].unique().astype(int).tolist())
    dates = read_active_split_dates(splits_path, args.fold)
    scoreable = panel["is_scoreable"].astype(bool).to_numpy()
    eligible = panel["is_eligible"].astype(bool).to_numpy()
    train_positions = np.flatnonzero(scoreable & np.isin(panel["Date"], dates["train"]))
    valid_positions = np.flatnonzero(scoreable & np.isin(panel["Date"], dates["validation"]))
    test_positions = np.flatnonzero(eligible & np.isin(panel["Date"], dates["test"]))
    base_pack = data_pack_from_panel(
        panel,
        manifest.columns,
        target_column=label.target_column,
        raw_return_column=label.raw_column,
    )

    output.mkdir(parents=True, exist_ok=True)
    if args.model in SEQUENCE_TORCH_MODELS:
        calendar = np.asarray(all_dates, dtype=np.int64)
        train_data = LazySequenceDataset(
            base_pack,
            SEQUENCE_LENGTH,
            calendar_dates=calendar,
            target_positions=train_positions,
        )
        valid_data = LazySequenceDataset(
            base_pack,
            SEQUENCE_LENGTH,
            calendar_dates=calendar,
            target_positions=valid_positions,
        )
        test_data = LazySequenceDataset(
            base_pack,
            SEQUENCE_LENGTH,
            calendar_dates=calendar,
            target_positions=test_positions,
        )
    else:
        train_data = base_pack.subset(train_positions)
        valid_data = base_pack.subset(valid_positions)
        test_data = base_pack.subset(test_positions)

    model_override: dict[str, object] = {}
    if args.threads and args.model == "lgb":
        model_override["num_threads"] = args.threads
    elif args.threads and args.model == "xgb":
        model_override["nthread"] = args.threads

    if args.model in TREE_MODELS:
        model = build_model(args.model, config=model_override or None)
        model.fit(train_data, valid_data)
        checkpoint = (
            paths.checkpoint(args.protocol, args.fold, args.model)
            if paths and args.output is None
            else output / ("best_model.txt" if args.model == "lgb" else "best_model.json")
        )
        checkpoint = model.save_checkpoint(checkpoint)
        valid_predictions = model.predict(valid_data)
        test_predictions = model.predict(test_data)
        history = model.evals_result_
        result = {
            "model": args.model,
            "fold": args.fold,
            "best_iteration": model.best_iteration_,
            "best_daily_ic": model.best_score_,
            "checkpoint": str(checkpoint),
        }
    else:
        training_override: dict[str, object] = {}
        if args.epochs:
            training_override["num_epochs"] = args.epochs
        if args.batch_size:
            training_override["batch_size"] = args.batch_size
        checkpoint = (
            paths.checkpoint(args.protocol, args.fold, args.model)
            if paths and args.output is None
            else output / "best_model.pt"
        )
        trainer, fitted = train_torch_model(
            args.model,
            train_data,
            valid_data,
            input_dim=len(manifest.columns),
            training_config=training_override or None,
            checkpoint_path=checkpoint,
            device=args.device,
        )
        valid_predictions = trainer.predict(valid_data)
        test_predictions = trainer.predict(test_data)
        history = list(fitted.history)
        result = {
            "model": args.model,
            "fold": args.fold,
            "best_epoch": fitted.best_epoch,
            "best_daily_ic": fitted.best_score,
            "checkpoint": str(fitted.checkpoint_path),
        }

    def targets(data: object) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(data, LazySequenceDataset):
            positions = data.target_positions
            return base_pack.require_labels()[positions], base_pack.raw_returns[positions]
        return data.require_labels(), data.raw_returns

    valid_target, valid_return = targets(valid_data)
    test_target, test_return = targets(test_data)
    validation_path = (
        paths.validation_predictions(args.protocol, args.fold, args.model)
        if paths and args.output is None
        else output / "validation_predictions.parquet"
    )
    test_path = (
        paths.test_predictions(args.protocol, args.fold, args.model)
        if paths and args.output is None
        else output / "test_predictions.parquet"
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(
        valid_data,
        valid_predictions,
        model=args.model,
        fold=args.fold,
        target=valid_target,
        returns=valid_return,
    ).to_parquet(validation_path, index=False)
    prediction_frame(
        test_data,
        test_predictions,
        model=args.model,
        fold=args.fold,
        target=test_target,
        returns=test_return,
    ).to_parquet(test_path, index=False)
    if isinstance(history, list):
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    else:
        write_json(history, output / "training_history.json")
    write_json(result, output / "result.json")
    print(f"finished {args.protocol}/{args.fold}/{args.model}: {len(test_predictions):,} test scores")


if __name__ == "__main__":
    main()
