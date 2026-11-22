# Data Cleaning Report

Source file: `data\messy_orders.csv`

## Summary

| Metric | Value |
|---|---|
| Rows in | 17 |
| Rows out | 14 |
| Duplicate rows removed | 2 |
| Text values normalised | 5 |
| Missing customers filled with 'Unknown' | 1 |
| Rows dropped for missing price | 1 |
| Missing quantities imputed | 1 |
| Negative quantities corrected | 1 |
| Non-positive prices flagged | 2 |
| Future dates flagged | 1 |
| Price outliers flagged (IQR) | 1 |
| Dates that could not be parsed | 0 |

## Missing values

| Column | Before | After |
|---|---|---|
| customer | 1 | 0 |
| category | 0 | 0 |
| quantity | 1 | 0 |
| unit_price | 1 | 0 |
| country | 0 | 0 |

## Date formats detected

| Format | Rows |
|---|---|
| `%Y-%m-%d` | 14 |
| `%d/%m/%Y` | 1 |

IQR bounds applied to unit price: **-98.25** to **225.75**.
Rows outside these bounds are flagged, not removed.

## Decisions

- Ambiguous dates were resolved day-first (Egypt/Gulf convention).
- Missing prices were dropped because they cannot be recovered from the row.
- Missing quantities were imputed with the median of valid quantities.
- Negative quantities were converted to absolute values and kept.
- Rows with a non-positive price are kept but excluded from revenue.
- Future dates are flagged, not removed, because they may be real pre-orders.
