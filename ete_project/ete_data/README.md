# ETE Data

This directory contains the model-ready intraday feature data used by training and backtest.

## Daily Parquet Data

```text
ete_data/min_feature_z-score/
```

Each trading day is stored as one parquet file:

```text
YYYYMMDD.parquet
```

The same file contains identifiers, thirty-minute z-score feature columns, the training label, and the realized return column used by monitoring and backtest.

Required identifier columns:

```text
Date
Code
```

Feature prefix:

```text
M30
```

Training label column:

```text
oto_d1_vwap-5min_mad-3_z-score_soft
```

Monitoring and backtest return column:

```text
oto_d1_vwap-5min
```

## NPY Sequence Data

```text
ete_data/min_feature_z-score_npy/
```

This path is used by `GRU` and `Transformer`. The sequence loader expects npy batch directories and metadata files under this root.
