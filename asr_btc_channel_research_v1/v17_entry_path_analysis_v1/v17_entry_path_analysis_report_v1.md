# v17 Entry Path Risk Analysis v1

## Source

- Version: v17
- Channel mode: baseline
- Data: 2024-01-01 00:00:00 to 2026-05-01 15:45:00
- Source trades: `D:\workspace\20260325\asr_btc_channel_research_v1\output_channel_v4\v17\trades_channel_v4.csv`
- Path observation starts from the next 15m bar after the entry close

## Core Backtest Summary

- Total return: 103.33%
- Sharpe: 1.620
- Max drawdown: -19.09%
- Trade count: 754
- Win rate: 60.21%
- Profit factor: 1.921

## Entry Path Summary

- First move adverse rate: 5.84%
- First move favorable rate: 0.00%
- Hit adverse 0.5% sometime during trade: 62.07%
- Hit favorable 0.5% sometime during trade: 78.25%
- Avg max adverse excursion: 1.30%
- Median max adverse excursion: 0.75%
- Avg underwater time: 5.87 hours
- Avg underwater ratio while holding: 39.66%
- Avg max consecutive underwater time: 4.70 hours

## Profit Accuracy After Entry Horizons

| Horizon | Samples | Profitable rate | Adverse-close rate | Avg close return | Avg max adverse to horizon |
|---:|---:|---:|---:|---:|---:|
| 15 min | 754 | 53.32% | 46.68% | 0.015% | 0.258% |
| 30 min | 741 | 54.12% | 45.88% | 0.029% | 0.363% |
| 60 min | 695 | 54.10% | 45.90% | 0.042% | 0.494% |
| 120 min | 640 | 55.62% | 44.38% | 0.087% | 0.648% |
| 180 min | 599 | 54.92% | 45.08% | 0.092% | 0.778% |
| 360 min | 498 | 56.22% | 43.78% | 0.231% | 1.001% |
| 720 min | 390 | 56.92% | 43.08% | 0.363% | 1.244% |
| 1440 min | 239 | 56.07% | 43.93% | 0.508% | 1.651% |
| 2880 min | 46 | 100.00% | 0.00% | 2.586% | 0.676% |

## Losing Trade Carry Summary

- Losing trades: 300
- Avg losing hold time: 14.00 hours
- Median losing hold time: 13.50 hours
- Avg underwater time in losing trades: 10.43 hours
- Median underwater time in losing trades: 7.25 hours
- Avg max adverse excursion in losing trades: 2.23%
- Median max adverse excursion in losing trades: 1.95%
- Avg max consecutive underwater time in losing trades: 8.68 hours

## Adverse/Favorable Threshold Hit Rates

| Type | Threshold | Hit rate | Avg hit minutes | Samples |
|---|---:|---:|---:|---:|
| adverse_pct | 0.25% | 77.72% | 62.3 | 754 |
| adverse_pct | 0.50% | 62.07% | 121.3 | 754 |
| adverse_pct | 1.00% | 41.38% | 227.8 | 754 |
| adverse_pct | 2.00% | 22.55% | 476.1 | 754 |
| adverse_pct | 3.00% | 11.94% | 668.0 | 754 |
| favorable_pct | 0.25% | 90.45% | 96.8 | 754 |
| favorable_pct | 0.50% | 78.25% | 186.9 | 754 |
| favorable_pct | 1.00% | 51.46% | 337.8 | 754 |
| favorable_pct | 2.00% | 26.53% | 572.8 | 754 |
| favorable_pct | 3.00% | 17.51% | 934.5 | 754 |

## Output Files

- `trade_path_metrics_v1.csv`
- `horizon_trade_metrics_v1.csv`
- `horizon_summary_v1.csv`
- `threshold_summary_v1.csv`
- `trade_group_summary_v1.csv`
- `loss_carry_summary_v1.csv`
- `metric_quantiles_v1.csv`
- `monthly_equity_summary_v1.csv`
