# Polymarket Multi-Horizon Research v1

This directory tests whether ASR BTC channel v17 entry signals have enough fixed-horizon up/down prediction edge to support Polymarket-style binary betting.

## Scope

- Source data: `D:\workspace\20260325\data\BTCUSDT_15m.csv`
- Source strategy logic: `D:\workspace\20260325\asr_btc_channel_research_v1` v17
- Main horizons: 15m, 30m, 60m, 120m, 180m, 360m, 720m, 1440m
- Primary focus: 60m, 360m, 1440m

## Run

```powershell
python run_polymarket_multi_horizon_backtest_v1.py
```

## Outputs

Outputs are written to `output_polymarket_multi_horizon_v1`.
