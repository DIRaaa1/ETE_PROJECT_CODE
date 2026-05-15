# Daily Data Download

This folder contains the daily bar download notebook.

## File

```text
day_gen.ipynb
```

## Data Source

The notebook uses `baostock` to download daily stock bars.

## Output

The notebook writes one parquet file per trading day:

```text
data/data_download/tusahre_day_download/stock_day/YYYYMMDD.parquet
```

It also stores intermediate batch files under:

```text
data/data_download/tusahre_day_download/stock_day/cache_batches/
```

## Columns

The configured daily fields are:

```text
date, code, open, high, low, close, preclose, volume, amount, pctChg, turn, tradestatus, isST
```

The notebook normalizes stock codes to six-digit strings and writes daily parquet files sorted by code.

## Main Parameters

- `START_DATE`: first download date.
- `END_DATE`: last download date.
- `ADJUSTFLAG`: price adjustment mode.
- `BATCH_SIZE`: number of stocks per cache batch.
- `OVERWRITE`: whether existing parquet files should be replaced.

## Role in the Project

Daily downloaded bars can be used as raw material for factor generation, industry merging, and final daily training parquet files.
