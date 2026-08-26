from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cross_sectional_ml.training import (
    DataPack,
    LazySequenceDataset,
    TorchTrainer,
    compute_regression_metrics,
    daily_ic_mean,
    daily_mse_mean,
    daily_r2_mean,
    safe_corr,
)


def test_daily_metrics_are_equal_weighted_by_date() -> None:
    dates = np.repeat([1, 2], 3)
    labels = np.tile([1.0, 2.0, 3.0], 2)
    predictions = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0])
    assert daily_ic_mean(predictions, labels, dates) == pytest.approx(0.5)
    assert daily_mse_mean(predictions, labels, dates) == pytest.approx(29.0 / 6.0)
    assert daily_r2_mean(predictions, labels, dates) == pytest.approx(-6.25)
    assert compute_regression_metrics(predictions, labels, dates) == pytest.approx(
        {
            "DailyICMean": 0.5,
            "DailyMSEMean": 29.0 / 6.0,
            "DailyR2Mean": -6.25,
            "NumDates": 2,
            "NumRows": 6,
        }
    )


def test_safe_corr_preserves_tiny_but_ordered_scores_and_maps_constants_to_zero() -> None:
    assert safe_corr(np.array([1e-20, 2e-20, 3e-20]), np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert safe_corr(np.ones(3), np.arange(3)) == 0.0


def test_lazy_sequence_dataset_uses_full_same_stock_calendar_windows() -> None:
    calendar = np.arange(1, 13)
    a_dates = calendar
    b_dates = calendar[calendar != 5]
    dates = np.concatenate((a_dates, b_dates))
    symbols = np.concatenate((np.repeat("A", len(a_dates)), np.repeat("B", len(b_dates))))
    x = dates.astype(np.float32).reshape(-1, 1)
    pack = DataPack(x, dates.astype(np.float32), dates, symbols, ("date_value",))

    dataset = LazySequenceDataset(pack, target_dates=[12])
    assert dataset.pack.x is pack.x
    assert dataset.shape == (1, 10, 1)
    assert dataset.symbols.tolist() == ["A"]
    features, target = dataset[0]
    np.testing.assert_array_equal(features.numpy().reshape(-1), np.arange(3, 13))
    assert target.item() == 12.0


def test_data_pack_rejects_misaligned_arrays() -> None:
    with pytest.raises(ValueError, match="dates has 1 rows, expected 2"):
        DataPack(
            np.zeros((2, 1), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            np.array([1]),
            np.array(["A", "B"]),
            ("x",),
        )


def _torch_packs() -> tuple[DataPack, DataPack]:
    rng = np.random.default_rng(7)
    features = []
    labels = []
    dates = []
    symbols = []
    for day in range(8):
        signal = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        noise = rng.normal(scale=0.2, size=8).astype(np.float32)
        features.append(np.column_stack((signal, noise)))
        labels.append(signal)
        dates.extend([day] * 8)
        symbols.extend([f"S{index:02d}" for index in range(8)])
    pack = DataPack(
        np.vstack(features),
        np.concatenate(labels),
        np.asarray(dates),
        np.asarray(symbols),
        ("signal", "noise"),
    )
    return pack.subset(np.arange(40)), pack.subset(np.arange(40, 64))


def test_torch_daily_ic_early_stop_checkpoint_and_predict(tmp_path: Path) -> None:
    train, valid = _torch_packs()
    checkpoint = tmp_path / "best_model.pt"
    trainer = TorchTrainer(
        "mlp",
        input_dim=2,
        model_config={"hidden_dims": [8], "dropout": 0.0},
        training_config={
            "learning_rate": 0.02,
            "batch_size": 16,
            "predict_batch_size": 32,
            "num_epochs": 3,
            "early_stop_patience": 1,
        },
        checkpoint_path=checkpoint,
        device="cpu",
        seed=3,
    )
    result = trainer.fit(train, valid)
    predictions = trainer.predict(valid)
    restored = TorchTrainer.from_checkpoint(checkpoint).predict(valid)

    assert checkpoint.exists()
    assert 1 <= result.best_epoch <= 3
    assert np.isfinite(result.best_score)
    assert len(result.history) <= 3
    assert predictions.shape == (valid.num_rows,)
    np.testing.assert_allclose(restored, predictions, rtol=0.0, atol=0.0)

    feature_only = DataPack(
        valid.x,
        None,
        valid.dates,
        valid.symbols,
        valid.feature_names,
    )
    assert trainer.predict(feature_only).shape == (valid.num_rows,)
