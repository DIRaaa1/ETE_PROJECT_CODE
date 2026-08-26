from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cross_sectional_ml.features import load_feature_manifest
from cross_sectional_ml.io import normalize_panel_keys
from cross_sectional_ml.labels import get_label_spec
from cross_sectional_ml.models import (
    MODEL_NAMES,
    SEQUENCE_LENGTH,
    SEQUENCE_TORCH_MODELS,
    LightGBMRegressor,
    XGBoostRegressor,
)
from cross_sectional_ml.training import LazySequenceDataset, TorchTrainer
from cross_sectional_ml.universe import data_pack_from_panel
from cross_sectional_ml.utils import (
    ProjectPaths,
    prediction_frame,
    read_active_split_dates,
    read_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run checkpoint inference.")
    parser.add_argument("--paths", type=Path, help="Optional project path configuration.")
    parser.add_argument("--panel", type=Path, help="Prepared model panel.")
    parser.add_argument("--protocol", choices=("o2o", "de"), required=True)
    parser.add_argument("--splits", type=Path, help="Date-role table from build_splits.py.")
    parser.add_argument("--fold", choices=("F1", "F2"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.load(args.paths) if args.paths else None
    panel_path = args.panel or (paths.panel(args.protocol) if paths else None)
    splits_path = args.splits or (paths.splits if paths else None)
    checkpoint = args.checkpoint or (
        paths.checkpoint(args.protocol, args.fold, args.model) if paths else None
    )
    output_path = args.output or (
        paths.prediction(args.protocol, args.fold, args.model, args.split) if paths else None
    )
    if panel_path is None or splits_path is None or checkpoint is None or output_path is None:
        raise ValueError(
            "provide --panel, --splits, --checkpoint, and --output, or derive them with --paths"
        )

    panel = normalize_panel_keys(read_frame(panel_path))
    manifest = load_feature_manifest(args.protocol)
    label = get_label_spec(args.protocol)
    target_column = label.target_column if label.target_column in panel.columns else None
    return_column = label.raw_column if label.raw_column in panel.columns else None
    pack = data_pack_from_panel(
        panel,
        manifest.columns,
        target_column=target_column,
        raw_return_column=return_column,
    )
    target_dates = read_active_split_dates(splits_path, args.fold)[args.split]
    in_split = np.isin(pack.dates, target_dates)
    target_positions = (
        np.flatnonzero(panel["is_eligible"].astype(bool).to_numpy() & in_split)
        if "is_eligible" in panel.columns
        else np.flatnonzero(in_split)
    )

    if args.model == "lgb":
        model = LightGBMRegressor.load_checkpoint(checkpoint)
        data = pack.subset(target_positions)
        predictions = model.predict(data)
    elif args.model == "xgb":
        model = XGBoostRegressor.load_checkpoint(checkpoint)
        data = pack.subset(target_positions)
        predictions = model.predict(data)
    else:
        trainer = TorchTrainer.from_checkpoint(checkpoint, device=args.device)
        data = (
            LazySequenceDataset(
                pack,
                SEQUENCE_LENGTH,
                calendar_dates=np.unique(pack.dates),
                target_positions=target_positions,
            )
            if args.model in SEQUENCE_TORCH_MODELS
            else pack.subset(target_positions)
        )
        predictions = trainer.predict(data)

    if isinstance(data, LazySequenceDataset):
        positions = data.target_positions
        target = None if pack.y is None else pack.y[positions]
        returns = None if pack.raw_returns is None else pack.raw_returns[positions]
    else:
        target = data.y
        returns = data.raw_returns
    output = prediction_frame(
        data,
        predictions,
        model=args.model,
        fold=args.fold,
        target=target,
        returns=returns,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)
    print(f"wrote {len(output):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
