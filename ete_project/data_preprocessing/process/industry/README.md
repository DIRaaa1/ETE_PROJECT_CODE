# Industry Feature Processing

This folder contains local industry classification processing files for daily parquet data.

## Files

```text
zhongxin.ipynb
shenwan.ipynb
```

## Zhongxin Processing

The Zhongxin process reads daily parquet files and Zhongxin industry history files. It adds:

```text
ci_l1
ci_l2
ci_l3
```

Default paths:

```text
input daily data: data/data_gene/daily_data_gen/
industry source: data/data_preprocessing/process/industry/zhongxin_source/
output: data/data_preprocessing/process/industry/day_with_zhongxin/
cache: data/data_preprocessing/process/industry/cache/
```

## Shenwan Processing

The Shenwan process reads the Zhongxin-enriched daily files and Shenwan industry history files. It adds:

```text
sw_l1
sw_l2
sw_l3
```

Default paths:

```text
input daily data: data/data_preprocessing/process/industry/day_with_zhongxin/
industry source: data/data_preprocessing/process/industry/shenwan_source/
output: data/data_preprocessing/process/industry/day_with_industry/
cache: data/data_preprocessing/process/industry/cache/
```

## Merge Logic

For each stock and trade date, the process finds the latest industry segment whose `in_date` is not later than the trade date. The match is retained only when the trade date is before the segment `out_date`.

## Role in the Project

Industry features can be joined into daily training data and used as categorical or encoded factor inputs in later modeling work.
