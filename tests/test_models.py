from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cross_sectional_ml.models import (
    MODEL_CONFIGS,
    MODEL_NAMES,
    SELECTED_CANDIDATES,
    LightGBMRegressor,
    XGBoostRegressor,
    build_model,
    build_torch_model,
)
from cross_sectional_ml.training import DataPack


def test_selected_model_registry() -> None:
    assert tuple(MODEL_CONFIGS) == MODEL_NAMES
    assert MODEL_NAMES == ("lgb", "xgb", "mlp", "kan", "gru", "tcn", "transformer", "mamba")
    assert SELECTED_CANDIDATES == {
        "lgb": "lgb__a04",
        "xgb": "xgb__a03",
        "mlp": "mlp__a04",
        "kan": "kan__a05",
        "gru": "gru__a05",
        "tcn": "tcn__a02",
        "transformer": "transformer__a02",
        "mamba": "mamba__a05",
    }


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        ("mlp", {"hidden_dims": [8, 4], "dropout": 0.0}),
        (
            "kan",
            {
                "hidden_dims": [6],
                "dropout": 0.0,
                "grid_size": 3,
                "spline_order": 2,
            },
        ),
    ],
)
def test_tabular_torch_models_produce_one_score_per_row(name: str, overrides: dict[str, object]) -> None:
    model = build_torch_model(name, input_dim=5, config=overrides)
    output = model(torch.randn(7, 5))
    assert output.shape == (7,)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("name", "overrides"),
    [
        (
            "gru",
            {"hidden_dim": 8, "num_layers": 2, "head_hidden_dim": 4, "dropout": 0.0},
        ),
        (
            "tcn",
            {"channels": [6, 6], "head_hidden_dim": 4, "dropout": 0.0},
        ),
        (
            "transformer",
            {
                "d_model": 8,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
                "max_len": 16,
            },
        ),
    ],
)
def test_sequence_torch_models_accept_ten_day_windows(name: str, overrides: dict[str, object]) -> None:
    model = build_torch_model(name, input_dim=5, config=overrides, sequence_length=10)
    output = model(torch.randn(4, 10, 5))
    assert output.shape == (4,)
    assert torch.isfinite(output).all()


def test_tree_factory_does_not_import_native_libraries_until_fit() -> None:
    assert isinstance(build_model("lgb"), LightGBMRegressor)
    assert isinstance(build_model("xgb"), XGBoostRegressor)
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("unknown")


def _tree_packs() -> tuple[DataPack, DataPack]:
    rng = np.random.default_rng(11)
    features = []
    labels = []
    dates = []
    symbols = []
    for day in range(8):
        cross_section = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
        noise = rng.normal(scale=0.1, size=10).astype(np.float32)
        features.append(np.column_stack((cross_section, noise)))
        labels.append(cross_section + 0.05 * noise)
        dates.extend([day] * 10)
        symbols.extend([f"S{index:02d}" for index in range(10)])
    pack = DataPack(
        np.vstack(features),
        np.concatenate(labels),
        np.asarray(dates),
        np.asarray(symbols),
        ("signal", "noise"),
    )
    return pack.subset(np.arange(50)), pack.subset(np.arange(50, 80))


@pytest.mark.parametrize(
    ("name", "overrides", "filename"),
    [
        (
            "lgb",
            {
                "num_boost_round": 12,
                "early_stopping_rounds": 3,
                "num_leaves": 7,
                "max_depth": 3,
                "min_data_in_leaf": 1,
                "num_threads": 1,
            },
            "best_model.txt",
        ),
        (
            "xgb",
            {
                "num_boost_round": 12,
                "early_stopping_rounds": 3,
                "max_depth": 3,
                "min_child_weight": 1,
                "nthread": 1,
            },
            "best_model.json",
        ),
    ],
)
def test_tree_daily_ic_training_checkpoint_and_reload(
    name: str, overrides: dict[str, object], filename: str, tmp_path: Path
) -> None:
    train, valid = _tree_packs()
    model = build_model(name, config=overrides)
    model.fit(train, valid)
    before = model.predict(valid)
    checkpoint = model.save_checkpoint(tmp_path / filename)
    reloaded = type(model).load_checkpoint(checkpoint, config=overrides)
    after = reloaded.predict(valid)
    assert model.best_iteration_ is not None
    assert np.isfinite(model.best_score_)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-6)
