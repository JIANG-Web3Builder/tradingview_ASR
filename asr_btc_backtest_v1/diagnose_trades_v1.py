from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("output_v1")
VERSIONS = ["v7", "v8", "v9", "v10", "v11", "v12"]


def pct(x: float) -> float:
    return round(float(x) * 100.0, 2)


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
    all_rows = []
    for version in VERSIONS:
        trades = pd.read_csv(BASE / version / "trades_v1.csv")
        summary = pd.read_json(BASE / version / "summary_v1.json", typ="series")
        trades["version"] = version
        all_rows.append(trades)
        print(f"\n=== {version} summary ===")
        print(
            {
                "return_pct": pct(summary["total_return"]),
                "sharpe": round(float(summary["sharpe"]), 4),
                "max_dd_pct": pct(summary["max_drawdown"]),
                "trades": int(summary["trade_count"]),
            }
        )
        print("\nby side")
        print(agg_table(trades, ["side"]))
        print("\nby level")
        print(agg_table(trades, ["level_snapshot"]))
        print("\nworst exit reasons")
        print(agg_table(trades, ["exit_reason"]).head(14))

    all_trades = pd.concat(all_rows, ignore_index=True)
    v12 = all_trades[all_trades["version"] == "v12"].copy()
    print("\n=== v12 worst side x level x exit clusters ===")
    print(agg_table(v12, ["side", "level_snapshot", "exit_reason"]).head(24))
    missed = v12[(v12["pnl_abs"] < 0) & (v12["mfe_pct"] > 0.5)].copy()
    print("\n=== v12 losers that had MFE > 0.5% ===")
    print(
        {
            "count": int(len(missed)),
            "pnl": round(float(missed["pnl_abs"].sum()), 2),
            "avg_mfe_pct": round(float(missed["mfe_pct"].mean()), 4) if len(missed) else 0.0,
            "avg_mae_pct": round(float(missed["mae_pct"].mean()), 4) if len(missed) else 0.0,
        }
    )
    if len(missed):
        print(agg_table(missed, ["side", "level_snapshot", "exit_reason"]).head(24))

    output = BASE / "diagnostics_v1.txt"
    with output.open("w", encoding="utf-8") as f:
        for version in VERSIONS:
            trades = all_trades[all_trades["version"] == version]
            f.write(f"\n=== {version} by side ===\n")
            f.write(agg_table(trades, ["side"]).to_string())
            f.write(f"\n\n=== {version} by level ===\n")
            f.write(agg_table(trades, ["level_snapshot"]).to_string())
            f.write(f"\n\n=== {version} by exit_reason ===\n")
            f.write(agg_table(trades, ["exit_reason"]).to_string())
            f.write("\n")


if __name__ == "__main__":
    main()
