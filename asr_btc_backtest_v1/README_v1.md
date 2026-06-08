# ASR BTC Python Backtest v1

## Files

- `config_v1.py`: global constants and version feature flags
- `data_loader_v1.py`: 15m CSV loading and validation
- `indicators_v1.py`: Pine-style indicator calculations
- `engine_v1.py`: shared backtest state machine for `v7-v12`
- `run_backtests_v1.py`: batch runner
- `metrics_v1.py`: summary metric computation
- `plotting_v1.py`: equity charts
- `strategy_asr_v7_v1.py` ~ `strategy_asr_v12_v1.py`: version markers

## Data refresh

Run:

```powershell
python D:\workspace\20260325\data\download_btc_v1.py
```

## Backtest

Run:

```powershell
python D:\workspace\20260325\asr_btc_backtest_v1\run_backtests_v1.py
```

## Output

All artifacts are written to:

- `D:\workspace\20260325\asr_btc_backtest_v1\output_v1`
