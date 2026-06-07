# Standardization Notes

This folder keeps earlier standardization notes. The current local project version uses:

```text
data/data_preprocessing/process/mad_standardize.py
```

## Current Method

The current script supports:

- cross-sectional MAD standardization by `Date`;
- time-series MAD standardization by `Code`.

For each group and each selected numeric column, the script applies:

1. median calculation;
2. MAD calculation;
3. clipping to `median +/- 3 * 1.4826 * MAD`;
4. MAD z-score calculation.


## Example Commands

```powershell
python .\data\data_preprocessing\process\mad_standardize.py --mode cross_section --input-dir .\data\data_train\alpha_360F_day --output-dir .\data\data_train\alpha_360F_day_mad3
python .\data\data_preprocessing\process\mad_standardize.py --mode time_series --input-dir .\data\data_train\alpha_360F_day --output-dir .\data\data_train\alpha_360F_day_ts_mad3
python .\data\data_preprocessing\process\mad_standardize.py --mode cross_section --input-dir .\data\data_train\oto_vwap_5min --output-dir .\data\data_train\oto_vwap_5min_mad3 --columns oto_d1_vwap-5min --output-suffix _mad3
```

Use `--columns` to pass an explicit comma-separated feature list, or use `--feature-prefix` to select matching numeric columns.
