# ASR BTC Python Strategy Iteration Report v1

## Method

This iteration did not use grid search or best-parameter hunting.

Each new version tested one structural mechanism at a time:

- trade management
- position sizing
- entry quality
- stop/exit behavior

The benchmark data period is:

- `2024-01-01 00:00:00` to `2026-05-01 15:45:00`
- BTCUSDT 15m
- no missing 15m spacing detected

## Version Summary

| Version | Mechanism | Return | Sharpe | MaxDD | Profit Factor | Trades | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| v12 | Pine-port baseline | 43.20% | 0.765 | -27.39% | 1.291 | 574 | baseline |
| v13 | normal L1/L2/L3 MFE protection | 65.94% | 1.133 | -21.68% | 1.522 | 700 | effective |
| v14 | add level0 breakout/reversal MFE protection | 89.17% | 1.422 | -20.87% | 1.730 | 705 | effective |
| v15 | L1 reclaim-only entry filter | 76.27% | 1.384 | -16.73% | 1.888 | 408 | defensive, too restrictive |
| v16 | half-size low-quality L1 knife entries | 98.04% | 1.560 | -19.70% | 1.898 | 705 | effective |
| v17 | no-progress time stop | 103.33% | 1.620 | -19.09% | 1.921 | 754 | current main candidate |
| v18 | half-size low-progress L2 adds | 95.15% | 1.576 | -17.81% | 1.929 | 754 | defensive, not main candidate |

## Effective Mechanisms

## v13: MFE Protection for Normal Layer Trades

Problem observed:

- Many losing trades had previously achieved useful floating profit.
- These trades later reverted into `OutsideStop`, `TimeStop`, or stop exits.

Mechanism:

- If a normal L1/L2/L3 position had at least one `zoneOffset` of MFE and then reverted near breakeven, exit early.

Result:

- Return improved from `43.20%` to `65.94%`.
- Sharpe improved from `0.765` to `1.133`.
- MaxDD improved from `-27.39%` to `-21.68%`.

Decision:

- Keep.

## v14: MFE Protection for Level0 Breakout/Reversal Trades

Problem observed after v13:

- Remaining major losses shifted toward `TrendLong`, `TrendShort`, `RevLong`, and `RevShort` level0 trades.

Mechanism:

- Apply MFE protection to level0 breakout/reversal trades after their structural lock/min-hold constraints are satisfied.

Result:

- Return improved from `65.94%` to `89.17%`.
- Sharpe improved from `1.133` to `1.422`.
- MaxDD improved from `-21.68%` to `-20.87%`.

Decision:

- Keep.

## v16: Half-Size Low-Quality L1 Knife Entries

Problem observed after v14:

- A strict L1 reclaim-only filter improved drawdown but removed too many profitable opportunities.
- The better response was to keep the trades but reduce risk when L1 entered before reclaim.

Mechanism:

- If Long1 enters below `midLow`, use half L1 size.
- If Short1 enters above `midHigh`, use half L1 size.
- Keep all entries; only adjust initial risk exposure.

Result:

- Return improved from `89.17%` to `98.04%`.
- Sharpe improved from `1.422` to `1.560`.
- MaxDD improved from `-20.87%` to `-19.70%`.
- Trade count stayed the same as v14.

Decision:

- Keep.

## v17: No-Progress Time Stop

Problem observed after v16:

- Large remaining losses were caused by trades that failed to produce meaningful favorable movement and then stayed open too long.

Mechanism:

- If a normal layered trade has held for `timeStopBars`, remains unprofitable, and has not produced at least one `zoneOffset` of MFE, close it early.

Result:

- Return improved from `98.04%` to `103.33%`.
- Sharpe improved from `1.560` to `1.620`.
- MaxDD improved from `-19.70%` to `-19.09%`.
- Average loss improved from `-48.71` to `-40.42`.

Decision:

- Keep.
- Current main candidate.

## Defensive but Not Main Candidate

## v15: L1 Reclaim-Only Filter

Result:

- Return dropped from `89.17%` to `76.27%`.
- MaxDD improved from `-20.87%` to `-16.73%`.
- Trade count fell from `705` to `408`.

Interpretation:

- It improved risk quality but removed too many valid L1 opportunities.

Decision:

- Keep only as a defensive reference.
- Do not tune this filter further.

## v18: Half-Size Low-Progress L2 Adds

Result:

- Return dropped from `103.33%` to `95.15%`.
- Sharpe dropped from `1.620` to `1.576`.
- MaxDD improved from `-19.09%` to `-17.81%`.
- Profit factor slightly improved from `1.921` to `1.929`.

Interpretation:

- It is a defensive variant.
- It reduces drawdown and average loss, but sacrifices too much return and Sharpe.

Decision:

- Not main candidate.
- Do not tune the half-size ratio further.

## Current Candidate Ranking

1. `v17`: best main candidate by return + Sharpe + acceptable drawdown.
2. `v18`: defensive candidate for lower drawdown preference.
3. `v16`: strong candidate if simpler than v17 is preferred.
4. `v14`: strong base after MFE management.
5. `v15`: too restrictive, not preferred.

## Current Main Candidate

Use `v17` as the current main Python strategy candidate.

Artifacts:

- engine: `engine_v6.py`
- config: `config_v6.py`
- strategy marker: `strategy_asr_v17_v1.py`
- runner: `run_backtests_v6.py`
- output: `output_v6/v17`

## Next Scientific Checks

Before translating the final candidate back into Pine or calling it final, do:

- validate v17 on time-split subperiods
- compare v17 vs v18 equity curves visually
- inspect the remaining large losses in `output_v6/diagnostics_v6.txt`
- consider out-of-sample or walk-forward checks if more data becomes available

## Avoided Overfitting

The following were intentionally not done:

- no parameter grid search
- no best threshold hunting
- no optimization of half-size ratios
- no repeated tuning of the same mechanism after a weak result

Every accepted change was based on a structural loss cluster found in trade logs.
