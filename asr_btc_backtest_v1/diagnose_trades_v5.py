from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path("output_v5")
VERSIONS = ["v14", "v16"]


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
    rows = []
    for version in VERSIONS:
        trades = pd.read_csv(BASE / version / "trades_v5.csv")
        summary = pd.read_json(BASE / version / "summary_v5.json", typ="series")
        trades["version"] = version
        rows.append(trades)
        print(f"\n=== {version} summary ===")
        print(summary[["total_return", "sharpe", "max_drawdown", "trade_count", "win_rate", "profit_factor"]])
        print("\nworst exit reasons")
        print(agg_table(trades, ["exit_reason"]).head(18))
        print("\nside x level")
        print(agg_table(trades, ["side", "level_snapshot"]))

    all_trades = pd.concat(rows, ignore_index=True)
    v16 = all_trades[all_trades["version"] == "v16"].copy()
    print("\n=== v16 worst side x level x exit clusters ===")
    print(agg_table(v16, ["side", "level_snapshot", "exit_reason"]).head(30))

    low_mfe_losers = v16[(v16["pnl_abs"] < 0) & (v16["mfe_pct"] <= 0.5)].copy()
    print("\n=== v16 immediate/low-MFE losers ===")
    print({"count": int(len(low_mfe_losers)), "pnl": round(float(low_mfe_losers["pnl_abs"].sum()), 2)})
    if len(low_mfe_losers):
        print(agg_table(low_mfe_losers, ["side", "level_snapshot", "exit_reason"]).head(25))

    high_mfe_losers = v16[(v16["pnl_abs"] < 0) & (v16["mfe_pct"] > 0.5)].copy()
    print("\n=== v16 high-MFE losers ===")
    print({"count": int(len(high_mfe_losers)), "pnl": round(float(high_mfe_losers["pnl_abs"].sum()), 2)})
    if len(high_mfe_losers):
        print(agg_table(high_mfe_losers, ["side", "level_snapshot", "exit_reason"]).head(25))

    level0 = v16[v16["level_snapshot"] == 0].copy()
    print("\n=== v16 level0 by entry/exit ===")
    print(agg_table(level0, ["entry_id", "exit_reason"]) if len(level0) else "no level0 trades")

    normal = v16[v16["level_snapshot"].isin([1, 2, 3])].copy()
    print("\n=== v16 normal by side x level x exit ===")
    print(agg_table(normal, ["side", "level_snapshot", "exit_reason"]).head(35))

    output = BASE / "diagnostics_v5.txt"
    with output.open("w", encoding="utf-8") as f:
        f.write("=== v16 worst side x level x exit clusters ===\n")
        f.write(agg_table(v16, ["side", "level_snapshot", "exit_reason"]).to_string())
        f.write("\n\n=== v16 low-MFE losers ===\n")
        f.write((agg_table(low_mfe_losers, ["side", "level_snapshot", "exit_reason"]).to_string() if len(low_mfe_losers) else "none"))
        f.write("\n\n=== v16 high-MFE losers ===\n")
        f.write((agg_table(high_mfe_losers, ["side", "level_snapshot", "exit_reason"]).to_string() if len(high_mfe_losers) else "none"))
        f.write("\n\n=== v16 level0 by entry/exit ===\n")
        f.write((agg_table(level0, ["entry_id", "exit_reason"]).to_string() if len(level0) else "none"))


if __name__ == "__main__":
    main()
