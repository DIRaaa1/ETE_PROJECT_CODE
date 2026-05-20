# Thirty-Minute Z-Score NPY Sequence Data

This directory is the default npy sequence input root for `GRU` and `Transformer`.

```text
ete_data/min_feature_z-score_npy/
```

The path is configured in `train/scModelHyperParams.json`:

```json
{
  "npy_root": "ete_data/min_feature_z-score_npy"
}
```

## Input Contract

The sequence loader expects batch folders under this directory. Batch folders use the configured prefix:

```text
batch_
```

Each batch folder may contain:

```text
data.npy
columns.json
sample_index.csv
sample_index.parquet
```

The root may also contain:

```text
days.json
```

`columns.json` must include the feature columns, identifier columns, the training label column, and the monitor return column.

Required identifier columns:

```text
Date
Code
```

Training label column:

```text
oto_d1_vwap-5min_mad-3_z-score_soft
```

Monitoring return column:

```text
oto_d1_vwap-5min
```

The training code resolves this path relative to the project root and does not require any external absolute data path.
