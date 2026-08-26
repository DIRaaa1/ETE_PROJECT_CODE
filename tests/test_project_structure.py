from __future__ import annotations

import json
from pathlib import Path

from cross_sectional_ml.models import MODEL_CONFIGS, MODEL_NAMES
from cross_sectional_ml.models.lgb import LightGBMRegressor
from cross_sectional_ml.models.torch_models import TransformerRegressor as LegacyTransformerRegressor
from cross_sectional_ml.models.transformer import TransformerRegressor
from cross_sectional_ml.models.trees import LightGBMRegressor as LegacyLightGBMRegressor
from cross_sectional_ml.utils import MODEL_CONFIG_DIR, ProjectPaths


def test_each_registered_model_has_one_json_config() -> None:
    files = tuple(path.stem for path in sorted(MODEL_CONFIG_DIR.glob("*.json")))
    assert set(files) == set(MODEL_NAMES)
    for name in MODEL_NAMES:
        with (MODEL_CONFIG_DIR / f"{name}.json").open(encoding="utf-8") as handle:
            assert json.load(handle) == MODEL_CONFIGS[name]


def test_legacy_model_modules_reexport_split_implementations() -> None:
    assert LegacyLightGBMRegressor is LightGBMRegressor
    assert LegacyTransformerRegressor is TransformerRegressor


def test_project_paths_cover_training_inference_and_backtest() -> None:
    paths = ProjectPaths.load()
    root = Path(".")
    assert paths.root == root
    assert paths.panel("de") == root / "data/processed/panels/de.parquet"
    assert paths.splits == root / "data/processed/date_roles.csv"
    assert paths.train_dir("o2o", "F1", "lgb") == root / "outputs/o2o/F1/lgb"
    assert paths.validation_predictions("o2o", "F1", "lgb").name == "validation_predictions.parquet"
    assert paths.test_predictions("de", "F2", "mamba").name == "test_predictions.parquet"
    assert paths.checkpoint("o2o", "F1", "xgb").name == "best_model.json"
    assert paths.backtest("de") == root / "outputs/de/backtest"
