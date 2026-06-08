from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path("output_v3")
VERSIONS = ["v13", "v14"]


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
        trades = pd.read_csv(BASE / version / "trades_v3.csv")
        summary = pd.read_json(BASE / version / "summary_v3.json", typ="series")
        trades["version"] = version
        rows.append(trades)
        print(f"\n=== {version} summary ===")
        print(summary[["total_return", "sharpe", "max_drawdown", "trade_count", "win_rate", "profit_factor"]])
        print("\nworst exit reasons")
        print(agg_table(trades, ["exit_reason"]).head(18))
        print("\nside x level")
        print(agg_table(trades, ["side", "level_snapshot"]))

    all_trades = pd.concat(rows, ignore_index=True)
    v14 = all_trades[all_trades["version"] == "v14"].copy()
    print("\n=== v14 worst side x level x exit clusters ===")
    print(agg_table(v14, ["side", "level_snapshot", "exit_reason"]).head(30))
    protect = v14[v14["exit_reason"].str.startswith("MFEProtect", na=False)].copy()
    print("\n=== v14 protect contribution ===")
    print(agg_table(protect, ["side", "level_snapshot", "exit_reason"]) if len(protect) else "no protect trades")
    normal = v14[v14["level_snapshot"].isin([1, 2, 3])].copy()
    print("\n=== v14 normal entries by level x side ===")
    print(agg_table(normal, ["side", "level_snapshot"]))
    level0 = v14[v14["level_snapshot"] == 0].copy()
    print("\n=== v14 level0 by entry/exit ===")
    print(agg_table(level0, ["entry_id", "exit_reason"]) if len(level0) else "no level0 trades")

    missed = v14[(v14["pnl_abs"] < 0) & (v14["mfe_pct"] > 0.5)].copy()
    print("\n=== v14 losers with MFE > 0.5% ===")
    print({"count": int(len(missed)), "pnl": round(float(missed["pnl_abs"].sum()), 2)})
    if len(missed):
        print(agg_table(missed, ["side", "level_snapshot", "exit_reason"]).head(25))

    low_mfe_losers = v14[(v14["pnl_abs"] < 0) & (v14["mfe_pct"] <= 0.5)].copy()
    print("\n=== v14 immediate/low-MFE losers ===")
    print({"count": int(len(low_mfe_losers)), "pnl": round(float(low_mfe_losers["pnl_abs"].sum()), 2)})
    if len(low_mfe_losers):
        print(agg_table(low_mfe_losers, ["side", "level_snapshot", "exit_reason"]).head(25))

    output = BASE / "diagnostics_v3.txt"
    with output.open("w", encoding="utf-8") as f:
        f.write("=== v14 worst side x level x exit clusters ===\n")
        f.write(agg_table(v14, ["side", "level_snapshot", "exit_reason"]).to_string())
        f.write("\n\n=== v14 protect contribution ===\n")
        f.write((agg_table(protect, ["side", "level_snapshot", "exit_reason"]).to_string() if len(protect) else "no protect trades"))
        f.write("\n\n=== v14 level0 by entry/exit ===\n")
        f.write((agg_table(level0, ["entry_id", "exit_reason"]).to_string() if len(level0) else "no level0 trades"))


if __name__ == "__main__":
    main()
