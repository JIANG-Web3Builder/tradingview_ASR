from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("output_v2")
VERSIONS = ["v12", "v13"]


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
        trades = pd.read_csv(BASE / version / "trades_v2.csv")
        summary = pd.read_json(BASE / version / "summary_v2.json", typ="series")
        trades["version"] = version
        rows.append(trades)
        print(f"\n=== {version} ===")
        print(summary[["total_return", "sharpe", "max_drawdown", "trade_count", "win_rate", "profit_factor"]])
        print("\nby exit_reason")
        print(agg_table(trades, ["exit_reason"]).head(20))
        print("\nby side x level")
        print(agg_table(trades, ["side", "level_snapshot"]))

    all_trades = pd.concat(rows, ignore_index=True)
    v13 = all_trades[all_trades["version"] == "v13"].copy()
    print("\n=== v13 worst side x level x exit clusters ===")
    print(agg_table(v13, ["side", "level_snapshot", "exit_reason"]).head(30))
    protect = v13[v13["exit_reason"].str.startswith("MFEProtect", na=False)].copy()
    print("\n=== v13 MFEProtect contribution ===")
    print(agg_table(protect, ["side", "level_snapshot", "exit_reason"]) if len(protect) else "no MFEProtect trades")
    missed = v13[(v13["pnl_abs"] < 0) & (v13["mfe_pct"] > 0.5)].copy()
    print("\n=== v13 losers that had MFE > 0.5% ===")
    print({"count": int(len(missed)), "pnl": round(float(missed["pnl_abs"].sum()), 2)})
    if len(missed):
        print(agg_table(missed, ["side", "level_snapshot", "exit_reason"]).head(30))

    output = BASE / "diagnostics_v2.txt"
    with output.open("w", encoding="utf-8") as f:
        for version in VERSIONS:
            trades = all_trades[all_trades["version"] == version]
            f.write(f"\n=== {version} by exit_reason ===\n")
            f.write(agg_table(trades, ["exit_reason"]).to_string())
            f.write(f"\n\n=== {version} by side x level x exit ===\n")
            f.write(agg_table(trades, ["side", "level_snapshot", "exit_reason"]).to_string())
            f.write("\n")


if __name__ == "__main__":
    main()
