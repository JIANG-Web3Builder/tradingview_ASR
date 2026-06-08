from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader_channel_v2 import load_15m_data
from indicators_channel_v2 import compute_indicators


OUTPUT_DIR = Path("output_channel_v2")
MODES = {
    "v17": "baseline",
    "v17_ch1": "atr_regime",
    "v17_ch2": "swing_range",
}


def indicator_stats(df: pd.DataFrame, version: str, mode: str) -> dict:
    valid = df.dropna(subset=["smoothRes", "smoothSup", "smoothMid", "midHigh", "midLow", "zoneOffset", "dynamicOffset"]).copy()
    width_pct = valid["channelWidth"] / valid["close"] * 100.0
    offset_pct = valid["dynamicOffset"] / valid["close"] * 100.0
    rows = {
        "version": version,
        "channel_mode": mode,
        "bars": int(len(valid)),
        "width_pct_mean": float(width_pct.mean()),
        "width_pct_median": float(width_pct.median()),
        "width_pct_p10": float(width_pct.quantile(0.10)),
        "width_pct_p90": float(width_pct.quantile(0.90)),
        "offset_pct_mean": float(offset_pct.mean()),
        "offset_pct_median": float(offset_pct.median()),
        "long_l1_touch_pct": float((valid["close"] <= valid["midLow"] + valid["zoneOffset"]).mean() * 100.0),
        "long_l2_touch_pct": float((valid["close"] <= valid["smoothSup"] + valid["zoneOffset"]).mean() * 100.0),
        "long_l3_touch_pct": float((valid["close"] <= valid["downerLine"] + valid["zoneOffset"]).mean() * 100.0),
        "short_l1_touch_pct": float((valid["close"] >= valid["midHigh"] - valid["zoneOffset"]).mean() * 100.0),
        "short_l2_touch_pct": float((valid["close"] >= valid["smoothRes"] - valid["zoneOffset"]).mean() * 100.0),
        "short_l3_touch_pct": float((valid["close"] >= valid["uperLine"] - valid["zoneOffset"]).mean() * 100.0),
        "breakout_up_touch_pct": float((valid["close"] > valid["superbuy"]).mean() * 100.0),
        "breakout_down_touch_pct": float((valid["close"] < valid["supersell"]).mean() * 100.0),
    }
    return rows


def monthly_indicator_stats(df: pd.DataFrame, version: str, mode: str) -> pd.DataFrame:
    valid = df.dropna(subset=["smoothRes", "smoothSup", "smoothMid", "midHigh", "midLow", "zoneOffset", "dynamicOffset"]).copy()
    valid["month"] = valid["open_time"].dt.to_period("M").astype(str)
    valid["width_pct"] = valid["channelWidth"] / valid["close"] * 100.0
    valid["offset_pct"] = valid["dynamicOffset"] / valid["close"] * 100.0
    valid["l1_touch"] = (valid["close"] <= valid["midLow"] + valid["zoneOffset"]) | (valid["close"] >= valid["midHigh"] - valid["zoneOffset"])
    valid["l2_touch"] = (valid["close"] <= valid["smoothSup"] + valid["zoneOffset"]) | (valid["close"] >= valid["smoothRes"] - valid["zoneOffset"])
    valid["breakout_touch"] = (valid["close"] > valid["superbuy"]) | (valid["close"] < valid["supersell"])
    monthly = (
        valid.groupby("month")
        .agg(
            width_pct_median=("width_pct", "median"),
            offset_pct_median=("offset_pct", "median"),
            l1_touch_pct=("l1_touch", lambda x: x.mean() * 100.0),
            l2_touch_pct=("l2_touch", lambda x: x.mean() * 100.0),
            breakout_touch_pct=("breakout_touch", lambda x: x.mean() * 100.0),
        )
        .reset_index()
    )
    monthly.insert(0, "version", version)
    monthly.insert(1, "channel_mode", mode)
    return monthly


def trade_stats(version: str) -> tuple[dict, pd.DataFrame]:
    trades = pd.read_csv(OUTPUT_DIR / version / "trades_channel_v2.csv")
    summary = pd.read_json(OUTPUT_DIR / version / "summary_channel_v2.json", typ="series")
    row = {
        "version": version,
        "channel_mode": str(summary["channel_mode"]),
        "total_return_pct": float(summary["total_return"]) * 100.0,
        "sharpe": float(summary["sharpe"]),
        "max_drawdown_pct": float(summary["max_drawdown"]) * 100.0,
        "trade_count": int(summary["trade_count"]),
        "profit_factor": float(summary["profit_factor"]),
        "avg_trade_pnl": float(summary["avg_trade_pnl"]),
        "avg_win": float(summary["avg_win"]),
        "avg_loss": float(summary["avg_loss"]),
    }
    exit_stats = (
        trades.groupby(["exit_reason", "level_snapshot"])
        .agg(
            trades=("pnl_abs", "size"),
            pnl=("pnl_abs", "sum"),
            avg=("pnl_abs", "mean"),
            wr=("pnl_abs", lambda x: (x > 0).mean()),
            avg_mfe=("mfe_pct", "mean"),
            avg_mae=("mae_pct", "mean"),
        )
        .reset_index()
    )
    exit_stats.insert(0, "version", version)
    return row, exit_stats


def main() -> None:
    raw = load_15m_data()
    indicator_rows = []
    monthly_rows = []
    trade_rows = []
    exit_rows = []
    for version, mode in MODES.items():
        indicators = compute_indicators(raw, {"channel_mode": mode})
        indicator_rows.append(indicator_stats(indicators, version, mode))
        monthly_rows.append(monthly_indicator_stats(indicators, version, mode))
        trade_row, exit_stats = trade_stats(version)
        trade_rows.append(trade_row)
        exit_rows.append(exit_stats)

    indicator_df = pd.DataFrame(indicator_rows)
    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    trade_df = pd.DataFrame(trade_rows)
    exit_df = pd.concat(exit_rows, ignore_index=True)

    indicator_df.to_csv(OUTPUT_DIR / "channel_indicator_diagnostics_v1.csv", index=False)
    monthly_df.to_csv(OUTPUT_DIR / "channel_monthly_diagnostics_v1.csv", index=False)
    trade_df.to_csv(OUTPUT_DIR / "channel_trade_summary_v1.csv", index=False)
    exit_df.to_csv(OUTPUT_DIR / "channel_exit_diagnostics_v1.csv", index=False)

    print("=== Trade summary ===")
    print(trade_df.round(4).to_string(index=False))
    print("\n=== Indicator summary ===")
    print(indicator_df.round(4).to_string(index=False))
    print("\n=== Worst exit clusters ===")
    print(exit_df.sort_values("pnl").groupby("version").head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
