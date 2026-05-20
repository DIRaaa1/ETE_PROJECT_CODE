# Thirty-Minute Intraday Z-Score Features

The full thirty-minute input block contains 80 per-bar fields across 8 intraday bars, so the complete intraday feature matrix provides 8 x 80 = 640 model input features before identifier columns such as date and ticker.

This directory documents the thirty-minute intraday feature layer used by the return prediction pipeline. The feature set is built from one-minute A-share market data and is intended for daily cross-sectional modelling. Each trading day is split into 8 thirty-minute bars, and each bar carries the same retained set of price, return, liquidity, activity, path, and distribution features.

The formulas below define the raw feature value before the final date-level z-score transform. Price-level fields are calculated from post-adjusted prices and divided by the stock's own post-adjusted close for the same trading day. Volume and amount fields that carry stock-size units are divided by same-day total volume or same-day total amount. Returns, ratios, positions, moments, and shape variables are already dimensionless before z-scoring.

## Output Contract

Each output column follows this pattern:

```text
M30_<FIELD_NAME>_<i>
```

where `i = 1, ..., 8` is the thirty-minute bar index within the trading day. For example, `M30_OPEN_1` is the scaled open price of the first thirty-minute bar, and `M30_RET_OC_8` is the open-to-close return of the last thirty-minute bar.

If all 80 per-bar fields are retained, the intraday block contains `80 * 8 = 640` feature columns, before identifier columns such as date and ticker.

## Z-Score Transform

For stock `s`, date `d`, bar `i`, and raw feature `x_{s,d,i}`, the stored z-score value is:

```text
z_{s,d,i} = safe_div(x_{s,d,i} - mean_{u in U_d}(x_{u,d,i}), std_{u in U_d}(x_{u,d,i}))
```

where `U_d` is the valid cross-sectional stock universe on date `d`. The same transform is applied separately to each output field and each thirty-minute bar.

## Symbol Conventions

Stock and date subscripts are omitted. All symbols refer to one stock on one trading day.

- `i`: thirty-minute bar index, `i = 1, ..., 8`.
- `B_i`: set of valid one-minute records inside bar `i`.
- `n_i`: number of valid one-minute records inside `B_i`.
- `q`: within-bar minute position, `q = 1, ..., n_i`.
- `O_i, H_i, L_i, C_i`: raw open, high, low, and close of bar `i`.
- `P_{i,q}`: one-minute close at position `q` in bar `i`.
- `h_{i,q}, l_{i,q}`: one-minute high and low at position `q` in bar `i`.
- `v_{i,q}`: one-minute volume at position `q` in bar `i`.
- `a_{i,q}`: one-minute amount at position `q` in bar `i`.
- `F_D`: daily adjustment factor.
- `X^*`: post-adjusted value. For example, `C_i^* = C_i F_D`.
- `V_i = sum_{q in B_i} v_{i,q}`: bar volume.
- `V_D = sum_i V_i`: total daily volume.
- `A_i = sum_{q in B_i} a_{i,q}`: bar amount.
- `A_D = sum_i A_i`: total daily amount.
- `A_i^* = A_i F_D`, `A_D^* = A_D F_D`: post-adjusted amount measures.
- `W_i = A_i / V_i`: raw bar VWAP.
- `W_i^* = W_i F_D`: post-adjusted bar VWAP.
- `C_D^*`: post-adjusted close of the last valid thirty-minute bar of the day.
- `r_{i,q} = P_{i,q}^* / P_{i,q-1}^* - 1`: one-minute return inside bar `i`.
- `N_i^r`: number of finite one-minute returns inside bar `i`.
- `safe_div(x, y)`: returns `x / y` only when `y` is finite and `abs(y) > eps`; otherwise the feature is left missing before z-scoring.

All divisions in the formulas below use safe division. Population variance and standard deviation are used for intrabar distribution fields unless stated otherwise.

## Formula Table

### 1. Price Level, Return, And Bar Shape

| Output field | Symbol | Formula | Definition |
|---|---|---|---|
| `M30_OPEN_i` | `Otilde_i` | `safe_div(O_i^*, C_D^*)` | Thirty-minute open price, post-adjusted and scaled by the same stock's adjusted day close. |
| `M30_CLOSE_i` | `Ctilde_i` | `safe_div(C_i^*, C_D^*)` | Thirty-minute close price, post-adjusted and scaled by the adjusted day close. |
| `M30_HIGH_i` | `Htilde_i` | `safe_div(H_i^*, C_D^*)` | Thirty-minute high price, post-adjusted and scaled by the adjusted day close. |
| `M30_LOW_i` | `Ltilde_i` | `safe_div(L_i^*, C_D^*)` | Thirty-minute low price, post-adjusted and scaled by the adjusted day close. |
| `M30_VWAP_i` | `Wtilde_i` | `safe_div(W_i^*, C_D^*)`, `W_i^* = safe_div(A_i, V_i) F_D` | Thirty-minute VWAP, post-adjusted and scaled by the adjusted day close. |
| `M30_RET_OC_i` | `r_i^OC` | `safe_div(C_i^*, O_i^*) - 1` | Open-to-close return of the current bar. |
| `M30_RET_CC_i` | `r_i^CC` | `safe_div(C_i^*, C_{i-1}^*) - 1` | Close-to-close return relative to the previous valid bar close. |
| `M30_GAP_i` | `gap_i` | `safe_div(O_i^*, C_{i-1}^*) - 1` | Opening gap relative to the previous valid bar close. |
| `M30_RET_VWAP_i` | `r_i^W` | `safe_div(W_i^*, W_{i-1}^*) - 1` | VWAP change relative to the previous valid bar VWAP. |
| `M30_RANGE_OPEN_i` | `range_i` | `safe_div(H_i^* - L_i^*, O_i^*)` | High-low range scaled by the bar open. |
| `M30_BODY_i` | `body_i` | `safe_div(C_i^* - O_i^*, O_i^*)` | Signed candlestick body. |
| `M30_ABS_BODY_i` | `absbody_i` | `safe_div(abs(C_i^* - O_i^*), O_i^*)` | Absolute candlestick body. |
| `M30_UPPER_SHADOW_i` | `ushadow_i` | `safe_div(max(H_i^* - max(O_i^*, C_i^*), 0), O_i^*)` | Upper shadow length scaled by the bar open. |
| `M30_LOWER_SHADOW_i` | `lshadow_i` | `safe_div(max(min(O_i^*, C_i^*) - L_i^*, 0), O_i^*)` | Lower shadow length scaled by the bar open. |
| `M30_BODY_RANGE_RATIO_i` | `bratio_i` | `safe_div(abs(C_i^* - O_i^*), H_i^* - L_i^*)` | Share of the high-low range occupied by the absolute body. |
| `M30_CLOSE_POS_i` | `cpos_i` | `safe_div(C_i^* - L_i^*, H_i^* - L_i^*)` | Close location inside the current bar high-low range. |
| `M30_VWAP_POS_i` | `wpos_i` | `safe_div(W_i^* - L_i^*, H_i^* - L_i^*)` | VWAP location inside the current bar high-low range. |
| `M30_CLOSE_VWAP_i` | `dev_i^{C,W}` | `safe_div(C_i^*, W_i^*) - 1` | Close price deviation from bar VWAP. |
| `M30_OPEN_VWAP_i` | `dev_i^{O,W}` | `safe_div(O_i^*, W_i^*) - 1` | Open price deviation from bar VWAP. |
| `M30_HIGH_VWAP_i` | `dev_i^{H,W}` | `safe_div(H_i^*, W_i^*) - 1` | High price deviation from bar VWAP. |
| `M30_LOW_VWAP_i` | `dev_i^{L,W}` | `safe_div(L_i^*, W_i^*) - 1` | Low price deviation from bar VWAP. |
| `M30_VWAP_OPEN_i` | `dev_i^{W,O}` | `safe_div(W_i^*, O_i^*) - 1` | VWAP deviation from bar open. |

For `i = 1`, the previous valid bar may be the last valid bar from the previous trading day when that history is available; otherwise previous-bar features are left missing before z-scoring.

### 2. Liquidity, Trading Activity, And Relative Impact

| Output field | Symbol | Formula | Definition |
|---|---|---|---|
| `M30_VOLUME_FIRST_i` | `vfirst_i^N` | `safe_div(v_{i,1}, V_D)` | First valid one-minute volume in the bar, scaled by total daily volume. |
| `M30_VOLUME_LAST_i` | `vlast_i^N` | `safe_div(v_{i,n_i}, V_D)` | Last valid one-minute volume in the bar, scaled by total daily volume. |
| `M30_VOLUME_MEAN_i` | `vmean_i^N` | `safe_div(mean_q(v_{i,q}), V_D)` | Mean one-minute volume in the bar, scaled by total daily volume. |
| `M30_AMOUNT_FIRST_i` | `afirst_i^N` | `safe_div(a_{i,1}^*, A_D^*) = safe_div(a_{i,1} F_D, A_D F_D)` | First valid one-minute amount in the bar, post-adjusted and scaled by adjusted total daily amount. |
| `M30_AMOUNT_LAST_i` | `alast_i^N` | `safe_div(a_{i,n_i}^*, A_D^*) = safe_div(a_{i,n_i} F_D, A_D F_D)` | Last valid one-minute amount in the bar, post-adjusted and scaled by adjusted total daily amount. |
| `M30_AMOUNT_MEAN_i` | `amean_i^N` | `safe_div(mean_q(a_{i,q}^*), A_D^*)` | Mean one-minute amount in the bar, post-adjusted and scaled by adjusted total daily amount. |
| `M30_VOL_CHG_i` | `chg_i^V` | `safe_div(V_i, V_{i-1}) - 1` | Bar volume change relative to the previous valid bar. |
| `M30_AMT_CHG_i` | `chg_i^A` | `safe_div(A_i, A_{i-1}) - 1` | Bar amount change relative to the previous valid bar. |
| `M30_VOL_SHARE_i` | `s_i^V` | `safe_div(V_i, V_D)` | Bar share of total daily volume. |
| `M30_AMT_SHARE_i` | `s_i^A` | `safe_div(A_i, A_D) = safe_div(A_i^*, A_D^*)` | Bar share of total daily amount. |
| `M30_RET_PER_AMT_i` | `impact_i^{r,A}` | `safe_div(r_i^OC, s_i^A)` | Open-to-close return per unit of daily amount share. |
| `M30_RET_PER_VOL_i` | `impact_i^{r,V}` | `safe_div(r_i^OC, s_i^V)` | Open-to-close return per unit of daily volume share. |
| `M30_ABS_RET_PER_AMT_i` | `impact_i^{absr,A}` | `safe_div(abs(r_i^OC), s_i^A)` | Absolute open-to-close return per unit of daily amount share. |
| `M30_ABS_RET_PER_VOL_i` | `impact_i^{absr,V}` | `safe_div(abs(r_i^OC), s_i^V)` | Absolute open-to-close return per unit of daily volume share. |
| `M30_RANGE_PER_AMT_i` | `impact_i^{range,A}` | `safe_div(range_i, s_i^A)` | High-low range per unit of daily amount share. |
| `M30_RANGE_PER_VOL_i` | `impact_i^{range,V}` | `safe_div(range_i, s_i^V)` | High-low range per unit of daily volume share. |
| `M30_UP_VOL_RATIO_i` | `uvr_i` | `safe_div(sum_{q:r_{i,q}>0} v_{i,q}, V_i)` | Share of bar volume traded during positive-return minutes. |
| `M30_DOWN_VOL_RATIO_i` | `dvr_i` | `safe_div(sum_{q:r_{i,q}<0} v_{i,q}, V_i)` | Share of bar volume traded during negative-return minutes. |
| `M30_NET_UP_VOL_RATIO_i` | `nuvr_i` | `safe_div(sum_{q:r_{i,q}>0} v_{i,q} - sum_{q:r_{i,q}<0} v_{i,q}, V_i)` | Net positive-minus-negative volume share. |
| `M30_UP_AMT_RATIO_i` | `uar_i` | `safe_div(sum_{q:r_{i,q}>0} a_{i,q}, A_i)` | Share of bar amount traded during positive-return minutes. |
| `M30_DOWN_AMT_RATIO_i` | `dar_i` | `safe_div(sum_{q:r_{i,q}<0} a_{i,q}, A_i)` | Share of bar amount traded during negative-return minutes. |
| `M30_NET_UP_AMT_RATIO_i` | `nuar_i` | `safe_div(sum_{q:r_{i,q}>0} a_{i,q} - sum_{q:r_{i,q}<0} a_{i,q}, A_i)` | Net positive-minus-negative amount share. |

### 3. Intrabar Return, Price Path, And Price Distribution

| Output field | Symbol | Formula | Definition |
|---|---|---|---|
| `M30_MIN_RET_STD_i` | `std_i^r` | `std_q(r_{i,q})` | Population standard deviation of one-minute returns inside the bar. |
| `M30_REALIZED_VOL_i` | `RV_i` | `sqrt(sum_q(r_{i,q}^2))` | Realized volatility inside the bar. |
| `M30_ABS_MIN_RET_MEAN_i` | `absrmean_i` | `mean_q(abs(r_{i,q}))` | Mean absolute one-minute return inside the bar. |
| `M30_MAX_MIN_RET_i` | `rmax_i` | `max_q(r_{i,q})` | Maximum one-minute return inside the bar. |
| `M30_MIN_MIN_RET_i` | `rmin_i` | `min_q(r_{i,q})` | Minimum one-minute return inside the bar. |
| `M30_UP_MIN_RATIO_i` | `p_i^up` | `count_q(r_{i,q} > 0) / N_i^r` | Share of positive one-minute returns. |
| `M30_DOWN_MIN_RATIO_i` | `p_i^down` | `count_q(r_{i,q} < 0) / N_i^r` | Share of negative one-minute returns. |
| `M30_FLAT_MIN_RATIO_i` | `p_i^flat` | `count_q(r_{i,q} = 0) / N_i^r` | Share of zero one-minute returns. |
| `M30_HIGH_TIME_POS_i` | `pos_i^H` | `first_argmax_q(h_{i,q}) / n_i` | First occurrence position of the intrabar high, scaled by valid minute count. |
| `M30_LOW_TIME_POS_i` | `pos_i^L` | `first_argmin_q(l_{i,q}) / n_i` | First occurrence position of the intrabar low, scaled by valid minute count. |
| `M30_HIGH_BEFORE_LOW_i` | `I_i^{H<L}` | `1[first_argmax_q(h_{i,q}) < first_argmin_q(l_{i,q})]` | Indicator that the high appears before the low. |
| `M30_INTRABAR_MAX_DRAWDOWN_i` | `MDD_i` | `min_q(safe_div(P_{i,q}^*, max_{u<=q} P_{i,u}^*) - 1)` | Maximum drawdown inside the bar based on one-minute close prices. |
| `M30_INTRABAR_MAX_REBOUND_i` | `MRB_i` | `max_q(safe_div(P_{i,q}^*, min_{u<=q} P_{i,u}^*) - 1)` | Maximum rebound inside the bar based on one-minute close prices. |
| `M30_PRICE_STD_OPEN_i` | `std_i^{P/O}` | `safe_div(std_q(P_{i,q}^*), O_i^*)` | Standard deviation of one-minute closes, scaled by the adjusted bar open. |
| `M30_PRICE_VAR_OPEN_i` | `var_i^{P/O}` | `safe_div(var_q(P_{i,q}^*), (O_i^*)^2)` | Variance of one-minute closes, scaled by the squared adjusted bar open. |
| `M30_PRICE_MAD_OPEN_i` | `mad_i^{P/O}` | `safe_div(mean_q(abs(P_{i,q}^* - mean_q(P_{i,q}^*))), O_i^*)` | Mean absolute deviation of one-minute closes, scaled by the adjusted bar open. |
| `M30_PRICE_DEV_VWAP_MEAN_i` | `meanDev_i^{P/W}` | `mean_q(safe_div(P_{i,q}^*, W_i^*) - 1)` | Mean one-minute close deviation from bar VWAP. |
| `M30_PRICE_DEV_VWAP_ABS_MEAN_i` | `meanAbsDev_i^{P/W}` | `mean_q(abs(safe_div(P_{i,q}^*, W_i^*) - 1))` | Mean absolute one-minute close deviation from bar VWAP. |
| `M30_PRICE_DEV_VWAP_STD_i` | `stdDev_i^{P/W}` | `std_q(safe_div(P_{i,q}^*, W_i^*) - 1)` | Standard deviation of one-minute close deviations from bar VWAP. |
| `M30_PRICE_MAX_DEV_OPEN_i` | `maxDev_i^{P/O}` | `max_q(safe_div(P_{i,q}^*, O_i^*) - 1)` | Maximum one-minute close deviation from bar open. |
| `M30_PRICE_MIN_DEV_OPEN_i` | `minDev_i^{P/O}` | `min_q(safe_div(P_{i,q}^*, O_i^*) - 1)` | Minimum one-minute close deviation from bar open. |
| `M30_PRICE_DEV_RANGE_OPEN_i` | `rangeDev_i^{P/O}` | `M30_PRICE_MAX_DEV_OPEN_i - M30_PRICE_MIN_DEV_OPEN_i` | Range of one-minute close deviations from bar open. |
| `M30_MIN_RET_VAR_i` | `var_i^r` | `E_q(r_{i,q}^2) - E_q(r_{i,q})^2` | Population variance of one-minute returns inside the bar. |
| `M30_MIN_RET_SKEW_i` | `skew_i^r` | `safe_div(E_q((r_{i,q} - mean_q(r_{i,q}))^3), (var_i^r)^(3/2))` | Skewness of one-minute returns. |
| `M30_MIN_RET_KURT_i` | `kurt_i^r` | `safe_div(E_q((r_{i,q} - mean_q(r_{i,q}))^4), (var_i^r)^2) - 3` | Excess kurtosis of one-minute returns. |
| `M30_UP_RET_MEAN_i` | `mean_i^{r+}` | `mean_q(r_{i,q} | r_{i,q} > 0)` | Mean return across positive-return minutes. |
| `M30_DOWN_RET_MEAN_i` | `mean_i^{r-}` | `mean_q(r_{i,q} | r_{i,q} < 0)` | Mean return across negative-return minutes. |
| `M30_UP_DOWN_RET_RATIO_i` | `ratio_i^{+/-}` | `safe_div(abs(M30_UP_RET_MEAN_i), abs(M30_DOWN_RET_MEAN_i))` | Positive-return mean magnitude relative to negative-return mean magnitude. |

### 4. Minute Volume Distribution

| Output field | Symbol | Formula | Definition |
|---|---|---|---|
| `M30_VOL_STD_i` | `std_i^{V,N}` | `safe_div(std_q(v_{i,q}), V_D)` | Standard deviation of one-minute volume, scaled by total daily volume. |
| `M30_VOL_VAR_i` | `var_i^{V,N}` | `safe_div(var_q(v_{i,q}), V_D^2)` | Variance of one-minute volume, scaled by squared total daily volume. |
| `M30_VOL_CV_i` | `cv_i^V` | `safe_div(std_q(v_{i,q}), mean_q(v_{i,q}))` | Coefficient of variation of one-minute volume. |
| `M30_VOL_MAD_MEAN_i` | `madmean_i^V` | `safe_div(mean_q(abs(v_{i,q} - mean_q(v_{i,q}))), mean_q(v_{i,q}))` | Mean absolute deviation of one-minute volume divided by mean volume. |
| `M30_VOL_SKEW_i` | `skew_i^V` | `safe_div(E_q((v_{i,q} - mean_q(v_{i,q}))^3), (var_q(v_{i,q}))^(3/2))` | Skewness of one-minute volume. |
| `M30_VOL_KURT_i` | `kurt_i^V` | `safe_div(E_q((v_{i,q} - mean_q(v_{i,q}))^4), (var_q(v_{i,q}))^2) - 3` | Excess kurtosis of one-minute volume. |
| `M30_VOL_SLOPE_NORM_i` | `slope_i^V` | `safe_div(slope(v_{i,q} ~ q), mean_q(v_{i,q}))` | Linear slope of one-minute volume over within-bar time, normalized by mean volume. |
| `M30_LAST5_FIRST5_VOL_RATIO_i` | `l5f5_i^V` | `safe_div(sum_{q in L5_i} v_{i,q}, sum_{q in F5_i} v_{i,q}) - 1` | Volume in the last five valid one-minute observations relative to the first five valid one-minute observations. |

where `F5_i = {q: q <= min(5, n_i)}` and `L5_i = {q: q > max(0, n_i - 5)}`.

## Notes On Scaling

- Price fields use post-adjusted prices, then scale by `C_D^*`. This removes the raw price level and keeps the cross-section comparable before z-scoring.
- Volume fields with raw size units use `V_D` as the denominator.
- Amount fields with raw size units use `A_D^*` as the denominator. Since the same daily adjustment factor is applied within the day, amount shares are unchanged by the adjustment factor.
- Return, ratio, position, moment, and coefficient-of-variation fields are dimensionless before z-scoring.
- `M30_VOLUME_SUM_i` and `M30_AMOUNT_SUM_i` are not listed separately. After daily scaling they duplicate `M30_VOL_SHARE_i` and `M30_AMT_SHARE_i`.
