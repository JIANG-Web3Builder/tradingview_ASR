from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config_polymarket_multi_horizon_v1 import BUY_PRICES, FEE_DRAGS, INITIAL_BANKROLL, OUTPUT_DIR, PRIMARY_HORIZON_BARS, SIGNAL_VERSION
from data_loader_polymarket_multi_horizon_v1 import load_15m_data, validate_15m_data
from market_simulator_polymarket_multi_horizon_v1 import add_pricing_results, build_binary_bets, build_concurrency_scenarios, build_equity_curve
from metrics_polymarket_multi_horizon_v1 import build_accuracy_by_group, build_price_sensitivity, build_variant_summary, save_summary_json
from plotting_polymarket_multi_horizon_v1 import plot_combined_equity_curves, plot_concurrency_comparison, plot_horizon_comparison
from signal_builder_polymarket_multi_horizon_v1 import build_v17_signals


def pct(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value * 100.0:.2f}%"


def signed_pct(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:+.2%}"


def build_overall_summary(raw_binary_bets: pd.DataFrame, binary_bets: pd.DataFrame, price_sensitivity: pd.DataFrame, validation: dict, signal_count: int) -> dict:
    primary_minutes = [bars * 15 for bars in PRIMARY_HORIZON_BARS]
    baseline = price_sensitivity[
        (price_sensitivity["variant"].isin(["all_signals", "long_only", "short_only"]))
        & (price_sensitivity["concurrency_mode"] == "independent")
        & (price_sensitivity["sample_split"] == "ALL")
        & (price_sensitivity["horizon_minutes"].isin(primary_minutes))
        & (price_sensitivity["buy_price"] == 0.5)
        & (price_sensitivity["fee_drag"] == 0.0)
    ].copy()
    best_oos = price_sensitivity[
        (price_sensitivity["sample_split"] == "OOS")
        & (price_sensitivity["concurrency_mode"] == "independent")
        & (price_sensitivity["buy_price"] == 0.5)
        & (price_sensitivity["fee_drag"] == 0.0)
        & (price_sensitivity["bet_count"] >= 20)
    ].sort_values("roi_on_stake", ascending=False).head(10)
    return {
        "version": SIGNAL_VERSION,
        "data_validation": validation,
        "signal_count": int(signal_count),
        "raw_binary_bet_rows": int(len(raw_binary_bets)),
        "scenario_binary_bet_rows": int(len(binary_bets)),
        "concurrency_modes": sorted(binary_bets["concurrency_mode"].dropna().unique().tolist()) if "concurrency_mode" in binary_bets.columns else [],
        "primary_horizon_minutes": primary_minutes,
        "baseline_rows": json.loads(baseline.to_json(orient="records")),
        "best_oos_price_0p50_fee_0_rows": json.loads(best_oos.to_json(orient="records")),
    }


def report_text(summary: dict, accuracy_by_group: pd.DataFrame, price_sensitivity: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Polymarket Multi-Horizon Backtest v1")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    validation = summary["data_validation"]
    lines.append(f"- Signal version: `{summary['version']}`")
    lines.append(f"- Data: {validation['start']} to {validation['end']}")
    lines.append(f"- 15m spacing bad count: {validation['bad_spacing_count']}")
    lines.append(f"- Signal count: {summary['signal_count']}")
    lines.append(f"- Raw fixed-horizon binary bet rows: {summary['raw_binary_bet_rows']}")
    lines.append(f"- Concurrency-expanded binary bet rows before pricing expansion: {summary['scenario_binary_bet_rows']}")
    lines.append(f"- Concurrency modes: {', '.join(summary['concurrency_modes'])}")
    lines.append("- Headline price tables use `independent` concurrency mode unless stated otherwise.")
    lines.append("- Fixed-horizon results settle at the target future close even if the original v17 strategy would have exited earlier.")
    lines.append("")
    lines.append("## Accuracy by Main Horizon")
    lines.append("")
    horizon_table = accuracy_by_group[accuracy_by_group["group_by"] == "side|horizon_minutes"].copy()
    horizon_table = horizon_table[horizon_table["horizon_minutes"].isin([15, 60, 360, 1440])]
    lines.append("| Side | Horizon | Samples | Win rate | Avg direction return | Median direction return |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in horizon_table.sort_values(["side", "horizon_minutes"]).itertuples(index=False):
        lines.append(f"| {row.side} | {int(row.horizon_minutes)}m | {int(row.sample_count)} | {row.win_rate * 100.0:.2f}% | {row.avg_direction_return_pct:.3f}% | {row.median_direction_return_pct:.3f}% |")
    lines.append("")
    lines.append("## Baseline Price Sensitivity at 0.50 Price and 0 Fee")
    lines.append("")
    baseline = price_sensitivity[
        (price_sensitivity["sample_split"] == "ALL")
        & (price_sensitivity["concurrency_mode"] == "independent")
        & (price_sensitivity["variant"].isin(["all_signals", "long_only", "short_only"]))
        & (price_sensitivity["horizon_minutes"].isin([15, 60, 360, 1440]))
        & (price_sensitivity["buy_price"] == 0.5)
        & (price_sensitivity["fee_drag"] == 0.0)
    ].copy()
    lines.append("| Variant | Horizon | Bets | Win rate | ROI on stake | Total PnL | Max DD | Profit factor |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in baseline.sort_values(["variant", "horizon_minutes"]).itertuples(index=False):
        lines.append(f"| {row.variant} | {int(row.horizon_minutes)}m | {int(row.bet_count)} | {row.win_rate * 100.0:.2f}% | {row.roi_on_stake * 100.0:.2f}% | {row.total_pnl:.2f} | {row.max_drawdown * 100.0:.2f}% | {row.profit_factor:.3f} |")
    lines.append("")
    lines.append("## OOS Best Cases at 0.50 Price and 0 Fee")
    lines.append("")
    best_oos = price_sensitivity[
        (price_sensitivity["sample_split"] == "OOS")
        & (price_sensitivity["concurrency_mode"] == "independent")
        & (price_sensitivity["buy_price"] == 0.5)
        & (price_sensitivity["fee_drag"] == 0.0)
        & (price_sensitivity["bet_count"] >= 20)
    ].sort_values("roi_on_stake", ascending=False).head(12)
    lines.append("| Variant | Horizon | Bets | Win rate | ROI | PnL | Max DD |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in best_oos.itertuples(index=False):
        lines.append(f"| {row.variant} | {int(row.horizon_minutes)}m | {int(row.bet_count)} | {row.win_rate * 100.0:.2f}% | {row.roi_on_stake * 100.0:.2f}% | {row.total_pnl:.2f} | {row.max_drawdown * 100.0:.2f}% |")
    lines.append("")
    lines.append("## Concurrency Mode Comparison")
    lines.append("")
    concurrency = price_sensitivity[
        (price_sensitivity["sample_split"] == "ALL")
        & (price_sensitivity["variant"] == "all_signals")
        & (price_sensitivity["horizon_minutes"].isin([60, 360, 1440]))
        & (price_sensitivity["buy_price"] == 0.5)
        & (price_sensitivity["fee_drag"] == 0.0)
    ].copy()
    lines.append("| Mode | Horizon | Bets | Win rate | ROI | Max DD |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in concurrency.sort_values(["concurrency_mode", "horizon_minutes"]).itertuples(index=False):
        lines.append(f"| {row.concurrency_mode} | {int(row.horizon_minutes)}m | {int(row.bet_count)} | {row.win_rate * 100.0:.2f}% | {row.roi_on_stake * 100.0:.2f}% | {row.max_drawdown * 100.0:.2f}% |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    long_1440 = baseline[(baseline["variant"] == "long_only") & (baseline["horizon_minutes"] == 1440)]
    short_1440 = baseline[(baseline["variant"] == "short_only") & (baseline["horizon_minutes"] == 1440)]
    all_360 = baseline[(baseline["variant"] == "all_signals") & (baseline["horizon_minutes"] == 360)]
    if not long_1440.empty:
        row = long_1440.iloc[0]
        lines.append(f"- Long-only 1440m baseline win rate is {row['win_rate'] * 100.0:.2f}%, so its rough breakeven buy price before fees is about {row['win_rate']:.3f}.")
    if not short_1440.empty:
        row = short_1440.iloc[0]
        lines.append(f"- Short-only 1440m baseline win rate is {row['win_rate'] * 100.0:.2f}%, confirming that shorts should not be blindly held long.")
    if not all_360.empty:
        row = all_360.iloc[0]
        lines.append(f"- All-signal 360m baseline ROI at 0.50 price is {row['roi_on_stake'] * 100.0:.2f}%, before real order-book frictions.")
    lines.append("- These results use synthetic buy prices; real Polymarket profitability requires historical fill prices or conservative price/fee sensitivity.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `signals_v1.csv`")
    lines.append("- `binary_bets_raw_multi_horizon_v1.csv`")
    lines.append("- `binary_bets_multi_horizon_v1.csv`")
    lines.append("- `priced_bets_multi_horizon_v1.csv`")
    lines.append("- `accuracy_by_group_multi_horizon_v1.csv`")
    lines.append("- `price_sensitivity_multi_horizon_v1.csv`")
    lines.append("- `variant_summary_multi_horizon_v1.csv`")
    lines.append("- `horizon_comparison_v1.png`")
    lines.append("- `combined_equity_curves_v1.png`")
    lines.append("- `concurrency_comparison_v1.png`")
    lines.append("- `summary_polymarket_multi_horizon_v1.json`")
    return "\n".join(lines)


def cleanup_old_plot_files() -> None:
    for pattern in ["equity_curve_*_price_*_v1.png", "equity_curve_all_signals_360m_base_v1.png"]:
        for path in OUTPUT_DIR.glob(pattern):
            path.unlink(missing_ok=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_plot_files()
    prices = load_15m_data()
    validation = validate_15m_data(prices)
    (OUTPUT_DIR / "data_validation_polymarket_multi_horizon_v1.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    signals, indicator_data, source_equity = build_v17_signals(prices)
    signals.to_csv(OUTPUT_DIR / "signals_v1.csv", index=False)
    source_equity.to_csv(OUTPUT_DIR / "source_equity_v17_v1.csv", index=False)

    raw_binary_bets = build_binary_bets(signals, indicator_data)
    raw_binary_bets.to_csv(OUTPUT_DIR / "binary_bets_raw_multi_horizon_v1.csv", index=False)
    binary_bets = build_concurrency_scenarios(raw_binary_bets)
    binary_bets.to_csv(OUTPUT_DIR / "binary_bets_multi_horizon_v1.csv", index=False)

    priced_bets = add_pricing_results(binary_bets, BUY_PRICES, FEE_DRAGS)
    priced_bets.to_csv(OUTPUT_DIR / "priced_bets_multi_horizon_v1.csv", index=False)

    accuracy_by_group = build_accuracy_by_group(raw_binary_bets)
    accuracy_by_group.to_csv(OUTPUT_DIR / "accuracy_by_group_multi_horizon_v1.csv", index=False)

    price_sensitivity = build_price_sensitivity(priced_bets)
    price_sensitivity.to_csv(OUTPUT_DIR / "price_sensitivity_multi_horizon_v1.csv", index=False)

    variant_summary = build_variant_summary(priced_bets)
    variant_summary.to_csv(OUTPUT_DIR / "variant_summary_multi_horizon_v1.csv", index=False)

    plot_horizon_comparison(price_sensitivity, OUTPUT_DIR / "horizon_comparison_v1.png")
    plot_combined_equity_curves(priced_bets, OUTPUT_DIR / "combined_equity_curves_v1.png")
    plot_concurrency_comparison(price_sensitivity, OUTPUT_DIR / "concurrency_comparison_v1.png")

    base_curve_input = priced_bets[
        (priced_bets["horizon_minutes"] == 360)
        & (priced_bets["buy_price"] == 0.5)
        & (priced_bets["fee_drag"] == 0.0)
        & (priced_bets["concurrency_mode"] == "independent")
    ].copy()
    base_curve = build_equity_curve(base_curve_input, INITIAL_BANKROLL)
    base_curve.to_csv(OUTPUT_DIR / "equity_curve_all_signals_360m_base_v1.csv", index=False)

    summary = build_overall_summary(raw_binary_bets, binary_bets, price_sensitivity, validation, len(signals))
    save_summary_json(OUTPUT_DIR / "summary_polymarket_multi_horizon_v1.json", summary)

    report = report_text(summary, accuracy_by_group, price_sensitivity)
    (OUTPUT_DIR / "research_report_polymarket_multi_horizon_v1.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
