from __future__ import annotations

from pathlib import Path

import pandas as pd


TRADES_PATH = Path("output_v6/v17/trades_v6.csv")
EQUITY_PATH = Path("output_v6/v17/equity_curve_v6.csv")
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2024-06-30 23:59:59")


def agg_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols)
        .agg(
            trades=("pnl_abs", "size"),
            pnl=("pnl_abs", "sum"),
            avg=("pnl_abs", "mean"),
            wr=("pnl_abs", lambda x: (x > 0).mean()),
            avg_mfe=("mfe_pct", "mean"),
            avg_mae=("mae_pct", "mean"),
            bars=("bars_held", "mean"),
        )
        .sort_values("pnl")
        .round(4)
    )


def main() -> None:
    trades = pd.read_csv(TRADES_PATH)
    equity = pd.read_csv(EQUITY_PATH)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    equity["open_time"] = pd.to_datetime(equity["open_time"])

    seg_trades = trades[(trades["exit_time"] >= START) & (trades["exit_time"] <= END)].copy()
    seg_equity = equity[(equity["open_time"] >= START) & (equity["open_time"] <= END)].copy()
    seg_trades["exit_month"] = seg_trades["exit_time"].dt.to_period("M").astype(str)

    monthly = []
    for month, mdf in seg_equity.groupby(seg_equity["open_time"].dt.to_period("M")):
        eq = mdf["equity"].astype(float)
        month_trades = seg_trades[seg_trades["exit_month"] == str(month)]
        monthly.append(
            {
                "month": str(month),
                "return_pct": (eq.iloc[-1] / eq.iloc[0] - 1.0) * 100.0 if len(eq) > 1 else 0.0,
                "max_dd_pct": (eq / eq.cummax() - 1.0).min() * 100.0 if len(eq) > 1 else 0.0,
                "trades": len(month_trades),
                "pnl_abs": month_trades["pnl_abs"].sum(),
                "win_rate_pct": (month_trades["pnl_abs"] > 0).mean() * 100.0 if len(month_trades) else 0.0,
            }
        )

    print("=== v17 2024_H1 monthly ===")
    print(pd.DataFrame(monthly).round(3).to_string(index=False))
    print("\n=== v17 2024_H1 worst exit reasons ===")
    print(agg_table(seg_trades, ["exit_reason"]).head(25))
    print("\n=== v17 2024_H1 worst side x level x exit ===")
    print(agg_table(seg_trades, ["side", "level_snapshot", "exit_reason"]).head(35))
    print("\n=== v17 2024_H1 level0 entry/exit ===")
    level0 = seg_trades[seg_trades["level_snapshot"] == 0]
    print(agg_table(level0, ["entry_id", "exit_reason"]) if len(level0) else "no level0 trades")
    print("\n=== v17 2024_H1 low-MFE losers ===")
    low_mfe = seg_trades[(seg_trades["pnl_abs"] < 0) & (seg_trades["mfe_pct"] <= 0.5)]
    print({"count": int(len(low_mfe)), "pnl": round(float(low_mfe["pnl_abs"].sum()), 2)})
    print(agg_table(low_mfe, ["side", "level_snapshot", "exit_reason"]).head(30) if len(low_mfe) else "none")
    print("\n=== v17 2024_H1 high-MFE losers ===")
    high_mfe = seg_trades[(seg_trades["pnl_abs"] < 0) & (seg_trades["mfe_pct"] > 0.5)]
    print({"count": int(len(high_mfe)), "pnl": round(float(high_mfe["pnl_abs"].sum()), 2)})
    print(agg_table(high_mfe, ["side", "level_snapshot", "exit_reason"]).head(30) if len(high_mfe) else "none")

    output = Path("output_v6/v17_2024h1_diagnostics_v1.txt")
    with output.open("w", encoding="utf-8") as f:
        f.write("=== monthly ===\n")
        f.write(pd.DataFrame(monthly).round(4).to_string(index=False))
        f.write("\n\n=== worst side x level x exit ===\n")
        f.write(agg_table(seg_trades, ["side", "level_snapshot", "exit_reason"]).to_string())
        f.write("\n\n=== low-MFE losers ===\n")
        f.write((agg_table(low_mfe, ["side", "level_snapshot", "exit_reason"]).to_string() if len(low_mfe) else "none"))
        f.write("\n\n=== high-MFE losers ===\n")
        f.write((agg_table(high_mfe, ["side", "level_snapshot", "exit_reason"]).to_string() if len(high_mfe) else "none"))


if __name__ == "__main__":
    main()
