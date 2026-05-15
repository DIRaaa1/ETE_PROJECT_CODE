# Chinese A-Share Return Prediction Project

This project studies cross-sectional return prediction for Chinese A-share stocks with machine learning models. The workflow starts from market data collection, builds daily and minute-level training data, applies cross-sectional normalization to factors and labels, and trains local PyTorch models with simple rolling windows.

The current codebase is intentionally local and compact. It does not depend on server paths, distributed training, graph-model data, or external training utilities. All paths are resolved under this project directory.

## Project Direction

The target task is to predict future stock returns in the Chinese A-share universe. Each trading day is treated as a cross-section of stocks. The training data uses one parquet file per day for tabular daily factors and labels, and optional sequence data stored as npy files for time-series models.

The main modeling paths are:

- `MLP`: uses daily factor parquet files and label parquet files.
- `GRU`: uses sequence npy data with shape `[sample, sequence_length, feature]`.

The training pipeline uses rolling windows. For each rolling window, the model trains on a fixed historical period, validates on the next period, skips a configurable gap, and predicts the following test period.

## Main Structure

```text
project/
  code/
    config/
    model/
    rolling_train/
  data/
    data_download/
    data_gene/
    data_preprocessing/
    data_train/
```

## Data Flow

1. Daily market data can be created by `data/data_download/tusahre_day_download/day_gen.ipynb`.
2. Minute market data can be created by `data/data_download/tushare_min_download/min_gen.ipynb`.
3. Industry features can be merged by the notebooks under `data/data_preprocessing/process/industry`.
4. Daily factors can be standardized by `data/data_preprocessing/process/z_score/signal_z-score.ipynb`.
5. Labels can be standardized by `data/data_preprocessing/process/z_score/label_z-score.ipynb`.
6. Model training is launched from `code/rolling_train/RollingTrain.py`.

## Training Inputs

The default training configuration expects:

```text
data/data_train/alpha_360F_day/
data/data_train/oto_vwap_5min/
data/data_train/alpha_360F_day_npy/
```

The first two directories are expected to contain one parquet file per trading day, named as `YYYYMMDD.parquet`. The npy directory supports either daily npy files or batch directories with metadata.

## Running Training

```powershell
python .\code\rolling_train\RollingTrain.py --model MLP
python .\code\rolling_train\RollingTrain.py --model GRU
```

Outputs are written to:

```text
outputs/MLP/
outputs/GRU/
```

Each run creates a timestamped result directory containing copied configs, copied model code, `rolling_windows.json`, `predictions.csv`, and best model checkpoints.

## Documentation Map

- `code/README.md`: code-level overview.
- `code/rolling_train/README.md`: rolling training architecture.
- `code/model/README.md`: model implementation details.
- `code/config/README.md`: training and model configuration details.
- `data/README.md`: data layer overview.
- `data/data_download/README.md`: data download overview.
- `data/data_download/tusahre_day_download/README.md`: daily data download details.
- `data/data_download/tushare_min_download/README.md`: minute data download details.
- `data/data_preprocessing/README.md`: preprocessing overview.
- `data/data_preprocessing/process/industry/README.md`: industry feature merge details.
- `data/data_preprocessing/process/z_score/README.md`: factor and label standardization details.
- `data/data_train/README.md`: final training data contract.
