# Training Data Contract

This directory is the final data interface consumed by the training code.

## Expected Directories

The default train config points to:

```text
data/data_train/alpha_360F_day/
data/data_train/oto_vwap_5min/
data/data_train/alpha_360F_day_npy/
```

If your local directories use different names, update:

```text
code/rolling_train/Train.json
```

## Daily Factor Parquet

Daily factor files should use:

```text
alpha_360F_day/YYYYMMDD.parquet
```

The default required columns are:

```text
Date
Code
factor columns
```

The trainer selects feature columns by excluding the configured date, symbol, label, and monitor-return columns. If `feature_prefix` is set in `Train.json`, matching columns are prioritized.

## Daily Label Parquet

Daily label files should use:

```text
oto_vwap_5min/YYYYMMDD.parquet
```

The default label column is:

```text
oto_d1_vwap-5min_mad-3_z-score_soft
```

The default monitor-return column is:

```text
oto_d1_vwap-5min
```

## NPY Sequence Data

The GRU path reads sequence arrays with shape:

```text
[sample, sequence_length, feature]
```

Two layouts are supported.

Daily layout:

```text
alpha_360F_day_npy/
  columns.json
  YYYYMMDD.npy
```

Batch layout:

```text
alpha_360F_day_npy/
  batch_000/
    batch_000.npy
    columns.json
    days.json
    sample_index.parquet
```

`columns.json` defines the feature order of the last dimension. `days.json` maps each trading day to its row count inside the batch npy file. `sample_index.parquet` is optional and can provide symbol identifiers.

## Date Naming

Daily parquet and npy dates should use:

```text
YYYYMMDD
```

Only dates within `start_date` and `end_date` in `Train.json` are used.
