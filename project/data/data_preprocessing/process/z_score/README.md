# Cross-Sectional Standardization

This folder contains notebooks for factor and label standardization.

## Files

```text
signal_z-score.ipynb
label_z-score.ipynb
```

## Factor Standardization

`signal_z-score.ipynb` processes daily factor parquet files.

Default input:

```text
data/data_train/alpha_360F_day/
```

Default output:

```text
data/data_train/alpha_360F_day_zscore/
```

For each daily cross-section, numeric factor columns are transformed with:

1. median absolute deviation filtering;
2. z-score normalization;
3. soft truncation outside the main range;
4. optional final clipping.

The default suffix is:

```text
_mad3_zscore_soft
```

## Label Standardization

`label_z-score.ipynb` processes label parquet files.

Default input:

```text
data/data_train/oto_vwap_5min/
```

Default output:

```text
data/data_train/oto_vwap_5min_zscore/
```

For each daily cross-section, selected label columns are transformed with:

1. cross-sectional rank;
2. z-score normalization of ranks.

The default suffix is:

```text
_rank_zscore
```

## Stock Universe Filter

Both notebooks keep stocks with code prefixes:

```text
0, 3, 6
```

## Role in the Project

The standardized factor and label files are intended to become the clean daily inputs for rolling model training.
