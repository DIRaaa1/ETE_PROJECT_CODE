# Chinese A-Share Return Prediction Project

This project contains the data contract, model training code, and backtest pipeline for daily cross-sectional return prediction on Chinese A-share data. All default paths are project-relative and resolve under the project root.

## Project Layout

```text
ete_project/
  backtest/
  basic_data/
    industry_data/
      zhongxin/
  ete_data/
    min_feature_z-score/
    min_feature_z-score_npy/
  model/
    config/
    deep_model/
    linear_model/
    tree_model/
  outputs/
  save_file/
  train/
  utils/
```

## Data Paths

Daily parquet input:

```text
ete_data/min_feature_z-score/
```

Each trading day is stored as one parquet file named:

```text
YYYYMMDD.parquet
```

The feature columns and label columns are stored in the same daily parquet file. Daily tabular models use this path for both `factor_dir` and `label_dir`.

Sequence npy input:

```text
ete_data/min_feature_z-score_npy/
```

GAT edge input:

```text
basic_data/industry_data/zhongxin/
```

## Supported Models

- `LightGBM`: tree model for daily tabular factors.
- `ELA`: elastic linear model for daily tabular factors.
- `MLP`: basic feed-forward model for daily tabular factors.
- `GRU`: basic sequence model for npy data.
- `Transformer`: basic sequence model for npy data.
- `GAT`: basic graph attention model for daily tabular factors with daily industry edge files.

## Training Configuration

The shared training configuration is:

```text
train/scModelHyperParams.json
```

The active default data paths are:

```json
{
  "factor_dir": "ete_data/min_feature_z-score",
  "label_dir": "ete_data/min_feature_z-score",
  "edge_pt_dir": "basic_data/industry_data/zhongxin",
  "npy_root": "ete_data/min_feature_z-score_npy"
}
```

Model hyperparameters are stored under:

```text
model/config/
```

## Run Training

```powershell
python .\train\scModelTrain.py --model LightGBM
python .\train\scModelTrain.py --model ELA
python .\train\scModelTrain.py --model MLP
python .\train\scModelTrain.py --model GRU
python .\train\scModelTrain.py --model Transformer
python .\train\scModelTrain.py --model GAT
```

Outputs are written under:

```text
outputs/<model>/
```

Each run creates a timestamped result directory containing the active training parameters, model hyperparameters, code snapshots, `rolling_windows.json`, checkpoints, monitor histories, and `predictions.csv`.

## Run Backtest

```powershell
python .\backtest\backtest.py
```

By default, the backtest reads predictions from:

```text
outputs/
```

and reads realized forward returns from:

```text
ete_data/min_feature_z-score/
```

Backtest outputs are written under:

```text
backtest/outputs/
```

## Documentation Map

- `ete_data/README.md`: data directories and input contracts.
- `ete_data/min_feature_z-score/README.md`: thirty-minute z-score parquet feature formulas.
- `ete_data/min_feature_z-score_npy/README.md`: npy sequence data contract.
- `basic_data/README.md`: static reference data and GAT edge location.
- `model/README.md`: retained model implementations and hyperparameter files.
- `train/README.md`: rolling training entry point and path rules.
- `backtest/README.md`: daily PnL markout backtest for model outputs.
- `save_file/README.md`: optional local archive area.
