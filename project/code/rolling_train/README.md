# Rolling Training

This directory contains the main local training entrypoint.

## Files

```text
rolling_train/
  RollingTrain.py
  Train.json
```

## Main Class

`RollingTrainer` coordinates the full training run. It is responsible for:

- resolving all relative paths under the project root;
- loading the train config and model hyperparameters;
- discovering available trading days;
- creating fixed rolling windows;
- loading parquet data for `MLP`;
- loading npy sequence data for `GRU`;
- training one model per rolling window;
- writing predictions and run metadata.

## Supported Models

`RollingTrainer` supports:

- `MLP`: reads daily factor parquet files and daily label parquet files.
- `GRU`: reads npy sequence data and its metadata.

No other model branches are active in the current local version.

## Rolling Logic

The rolling logic is fixed-window only. It uses:

- `fit_window_length`: number of available trading days in the training period.
- `valid_window_length`: number of available trading days in the validation period.
- `test_gap`: gap between validation end and test start.
- `step`: test window length and rolling step.

For each rolling window:

```text
train: fixed historical window
valid: immediately after train
gap: skipped days after valid
test: prediction window
```

The output rolling plan is saved as `rolling_windows.json`.

## MLP Data Loading

For `MLP`, the trainer expects:

```text
factor_dir/YYYYMMDD.parquet
label_dir/YYYYMMDD.parquet
```

The trainer intersects available dates from both directories. It reads factor columns from the factor parquet, merges labels by shared date and symbol columns when the factor and label files are separate, drops rows with missing required values, and builds tensors:

```text
x: [rows, features]
y: [rows]
row_dates: [rows]
symbols: [rows]
```

## GRU Data Loading

For `GRU`, the trainer expects npy sequence arrays with shape:

```text
[sample, sequence_length, feature]
```

Two storage layouts are supported:

- daily files: `npy_root/YYYYMMDD.npy` with a shared `columns.json`;
- batch directories: `npy_root/batch_x/batch_x.npy`, `columns.json`, and `days.json`.

The label is read from the last sequence step. Feature columns are selected from `columns.json`, excluding the date, symbol, label, and monitor-return columns.

## Commands

```powershell
python .\code\rolling_train\RollingTrain.py --model MLP
python .\code\rolling_train\RollingTrain.py --model GRU
```

Optional paths can be overridden:

```powershell
python .\code\rolling_train\RollingTrain.py --model MLP --train-config .\code\rolling_train\Train.json --model-config .\code\config\MLPHyperParams.json --save-dir .\outputs\MLP
```
