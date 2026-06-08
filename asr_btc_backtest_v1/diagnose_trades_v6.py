from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path("output_v6")
VERSIONS = ["v16", "v17"]


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
        trades = pd.read_csv(BASE / version / "trades_v6.csv")
        summary = pd.read_json(BASE / version / "summary_v6.json", typ="series")
        trades["version"] = version
        rows.append(trades)
        print(f"\n=== {version} summary ===")
        print(summary[["total_return", "sharpe", "max_drawdown", "trade_count", "win_rate", "profit_factor"]])
        print("\nworst exit reasons")
        print(agg_table(trades, ["exit_reason"]).head(18))
        print("\nside x level")
        print(agg_table(trades, ["side", "level_snapshot"]))

    all_trades = pd.concat(rows, ignore_index=True)
    v17 = all_trades[all_trades["version"] == "v17"].copy()
    print("\n=== v17 worst side x level x exit clusters ===")
    print(agg_table(v17, ["side", "level_snapshot", "exit_reason"]).head(35))

    level0 = v17[v17["level_snapshot"] == 0].copy()
    print("\n=== v17 level0 by entry/exit ===")
    print(agg_table(level0, ["entry_id", "exit_reason"]) if len(level0) else "no level0 trades")

    no_progress = v17[v17["exit_reason"].str.startswith("NoProgress", na=False)].copy()
    print("\n=== v17 NoProgress contribution ===")
    print(agg_table(no_progress, ["side", "level_snapshot", "exit_reason"]) if len(no_progress) else "no NoProgress trades")

    low_mfe_losers = v17[(v17["pnl_abs"] < 0) & (v17["mfe_pct"] <= 0.5)].copy()
    print("\n=== v17 immediate/low-MFE losers ===")
    print({"count": int(len(low_mfe_losers)), "pnl": round(float(low_mfe_losers["pnl_abs"].sum()), 2)})
    if len(low_mfe_losers):
        print(agg_table(low_mfe_losers, ["side", "level_snapshot", "exit_reason"]).head(30))

    high_mfe_losers = v17[(v17["pnl_abs"] < 0) & (v17["mfe_pct"] > 0.5)].copy()
    print("\n=== v17 high-MFE losers ===")
    print({"count": int(len(high_mfe_losers)), "pnl": round(float(high_mfe_losers["pnl_abs"].sum()), 2)})
    if len(high_mfe_losers):
        print(agg_table(high_mfe_losers, ["side", "level_snapshot", "exit_reason"]).head(30))

    winners = v17[v17["pnl_abs"] > 0].copy()
    print("\n=== v17 winner clusters ===")
    print(agg_table(winners, ["side", "level_snapshot", "exit_reason"]).tail(25))

    output = BASE / "diagnostics_v6.txt"
    with output.open("w", encoding="utf-8") as f:
        f.write("=== v17 worst side x level x exit clusters ===\n")
        f.write(agg_table(v17, ["side", "level_snapshot", "exit_reason"]).to_string())
        f.write("\n\n=== v17 NoProgress contribution ===\n")
        f.write((agg_table(no_progress, ["side", "level_snapshot", "exit_reason"]).to_string() if len(no_progress) else "none"))
        f.write("\n\n=== v17 level0 by entry/exit ===\n")
        f.write((agg_table(level0, ["entry_id", "exit_reason"]).to_string() if len(level0) else "none"))
        f.write("\n\n=== v17 low-MFE losers ===\n")
        f.write((agg_table(low_mfe_losers, ["side", "level_snapshot", "exit_reason"]).to_string() if len(low_mfe_losers) else "none"))


if __name__ == "__main__":
    main()
