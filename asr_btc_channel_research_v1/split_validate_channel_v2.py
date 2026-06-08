from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("output_channel_v4")
VERSIONS = ["v17", "v17_ch3", "v17_ch4"]
PERIODS = [
    ("2024_FULL", "2024-01-01", "2024-12-31 23:59:59"),
    ("2025_FULL", "2025-01-01", "2025-12-31 23:59:59"),
    ("2026_YTD", "2026-01-01", "2026-05-01 23:59:59"),
    ("2024_H1", "2024-01-01", "2024-06-30 23:59:59"),
    ("2024_H2", "2024-07-01", "2024-12-31 23:59:59"),
    ("2025_H1", "2025-01-01", "2025-06-30 23:59:59"),
    ("2025_H2", "2025-07-01", "2025-12-31 23:59:59"),
    ("2026_Q1", "2026-01-01", "2026-03-31 23:59:59"),
    ("2026_APR", "2026-04-01", "2026-05-01 23:59:59"),
]


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def period_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, start: str, end: str) -> dict:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    seg = equity_df[(equity_df["open_time"] >= start_ts) & (equity_df["open_time"] <= end_ts)].copy()
    trades = trades_df[(trades_df["exit_time"] >= start_ts) & (trades_df["exit_time"] <= end_ts)].copy()
    if len(seg) < 2:
        return {
            "return_pct": np.nan,
            "sharpe": np.nan,
            "max_dd_pct": np.nan,
            "trade_count": int(len(trades)),
            "win_rate_pct": np.nan,
            "pnl_abs": float(trades["pnl_abs"].sum()) if len(trades) else 0.0,
            "profit_factor": np.nan,
        }
    eq = seg["equity"].astype(float)
    ret = eq.pct_change().fillna(0.0)
    bars_per_year = 365.0 * 24.0 * 4.0
    sharpe = float((ret.mean() / ret.std(ddof=0)) * np.sqrt(bars_per_year)) if ret.std(ddof=0) > 0 else 0.0
    wins = trades[trades["pnl_abs"] > 0]
    losses = trades[trades["pnl_abs"] < 0]
    gross_profit = float(wins["pnl_abs"].sum()) if len(wins) else 0.0
    gross_loss = float(-losses["pnl_abs"].sum()) if len(losses) else 0.0
    return {
        "return_pct": float((eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0),
        "sharpe": sharpe,
        "max_dd_pct": max_drawdown(eq) * 100.0,
        "trade_count": int(len(trades)),
        "win_rate_pct": float((trades["pnl_abs"] > 0).mean() * 100.0) if len(trades) else 0.0,
        "pnl_abs": float(trades["pnl_abs"].sum()) if len(trades) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else 0.0,
    }


def main() -> None:
    rows = []
    for version in VERSIONS:
        equity_df = pd.read_csv(BASE / version / "equity_curve_channel_v4.csv")
        trades_df = pd.read_csv(BASE / version / "trades_channel_v4.csv")
        summary = pd.read_json(BASE / version / "summary_channel_v4.json", typ="series")
        equity_df["open_time"] = pd.to_datetime(equity_df["open_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        for period_name, start, end in PERIODS:
            rows.append(
                {
                    "version": version,
                    "channel_mode": str(summary["channel_mode"]),
                    "period": period_name,
                    **period_metrics(equity_df, trades_df, start, end),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(BASE / "split_validate_channel_v2.csv", index=False)
    print("=== Return % ===")
    print(result.pivot(index="period", columns="version", values="return_pct").round(2).to_string())
    print("\n=== Sharpe ===")
    print(result.pivot(index="period", columns="version", values="sharpe").round(3).to_string())
    print("\n=== MaxDD % ===")
    print(result.pivot(index="period", columns="version", values="max_dd_pct").round(2).to_string())
    print("\n=== Profit Factor ===")
    print(result.pivot(index="period", columns="version", values="profit_factor").round(3).to_string())


if __name__ == "__main__":
    main()
