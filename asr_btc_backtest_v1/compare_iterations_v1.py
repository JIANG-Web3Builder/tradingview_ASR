from __future__ import annotations

from pathlib import Path

import pandas as pd


RUNS = [
    ("v12", Path("output_v1/v12/summary_v1.json"), "v12 baseline from Pine port"),
    ("v13", Path("output_v2/v13/summary_v2.json"), "normal L1/L2/L3 MFE protection"),
    ("v14", Path("output_v3/v14/summary_v3.json"), "adds level0 breakout/reversal MFE protection"),
    ("v15", Path("output_v4/v15/summary_v4.json"), "defensive L1 reclaim-only entry filter"),
    ("v16", Path("output_v5/v16/summary_v5.json"), "half-size low-quality L1 knife entries"),
    ("v17", Path("output_v6/v17/summary_v6.json"), "adds no-progress time stop"),
    ("v18", Path("output_v7/v18/summary_v7.json"), "defensive low-progress L2 half-size adds"),
]


def main() -> None:
    rows = []
    for version, path, mechanism in RUNS:
        summary = pd.read_json(path, typ="series")
        rows.append(
            {
                "version": version,
                "mechanism": mechanism,
                "total_return_pct": float(summary["total_return"]) * 100.0,
                "sharpe": float(summary["sharpe"]),
                "max_drawdown_pct": float(summary["max_drawdown"]) * 100.0,
                "calmar": float(summary["calmar"]),
                "profit_factor": float(summary["profit_factor"]),
                "win_rate_pct": float(summary["win_rate"]) * 100.0,
                "trade_count": int(summary["trade_count"]),
                "avg_trade_pnl": float(summary["avg_trade_pnl"]),
                "avg_loss": float(summary["avg_loss"]),
            }
        )
    df = pd.DataFrame(rows)
    out = Path("iteration_comparison_v1.csv")
    df.to_csv(out, index=False)
    print(df.round(4).to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
