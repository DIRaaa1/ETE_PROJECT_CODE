# Daily Data Download

This folder contains the daily bar download notebook.

## File

```text
tushare_day.ipynb
```

## Data Source

The notebook uses `tushare` to download daily stock bars.

The notebook reads credentials from environment variables:

```text
TUSHARE_TOKEN
TUSHARE_HTTP_URL
```

`TUSHARE_HTTP_URL` is optional.

## Output

The notebook writes one parquet file per trading day:

```text
data/data_download/tusahre_day_download/stock_day/YYYYMMDD.parquet
```

## Columns

The base daily fields are:

```text
ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
```

The notebook also merges adjustment factors, daily basic fields, limit-up and limit-down prices, and stock ST status. It computes `vwap`, close-limit flags, intraday-hit flags, and open-limit flags. Before writing, `ts_code` is renamed to `code` and `trade_date` is renamed to `date`.

## Main Parameters

- `START_DATE`: first download date.
- `END_DATE`: last download date.
- `TUSHARE_TOKEN`: Tushare token from the environment.
- `TUSHARE_HTTP_URL`: optional Tushare-compatible endpoint.
- `START_DATE`: first download date.
- `END_DATE`: last download date.
- `LIMIT`: pagination size.

## Role in the Project

Daily downloaded bars can be used as raw material for factor generation, industry merging, and final daily training parquet files.
