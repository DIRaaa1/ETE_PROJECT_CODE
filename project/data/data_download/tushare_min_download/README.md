# Minute Data Download

This folder contains the minute bar download notebook.

## File

```text
tushare_min.ipynb
```

## Data Source

The notebook uses `tushare` with a token from:

```text
TUSHARE_TOKEN
```

An optional custom HTTP endpoint can be provided through:

```text
TUSHARE_HTTP_URL
```

## Inputs

The notebook can read:

```text
data/data_download/tushare_min_download/stock_list/
data/data_download/tusahre_day_download/stock_day/
```

Daily stock files provide the available trading-day calendar. Optional per-day stock-list JSON files provide the stock universe. Each JSON file can be either a list of codes or a dictionary with a `codes` list.

## Output

Minute files are written by year and stock:

```text
data/data_download/tushare_min_download/stock_minute/YYYY/<ts_code>.parquet
```

Each file contains normalized one-minute bars for one stock in that calendar year. Each year directory also contains a CSV log for missing chunks and request failures.

## Main Parameters

- `START_DATE`: first date included in the download.
- `END_DATE`: last date included in the download.
- `FREQ`: bar frequency.
- `ADJ`: adjustment mode.
- `LIMIT`: maximum rows per request.
- `CHUNK_TRADE_DAYS`: number of trade dates per request chunk.
- `RATE_LIMIT_PER_MIN`: request throttle.
- `ENABLE_OVERWRITE`: whether existing output files should be replaced.
- `ENABLE_PARALLEL`: whether stock downloads run concurrently.
- `STOCK_WORKERS`: worker count when parallel mode is enabled.

## Role in the Project

Minute data can be used to build intraday factors, generate sequence features, or support later npy conversion for time-series models.
