from __future__ import annotations

import pandas as pd

from config_multi_factor_v1 import INITIAL_CAPITAL, OUTPUT_DIR, VERSION_ORDER
from data_loader_multi_factor_v1 import load_15m_data, validate_15m_data
from engine_multi_factor_v1 import run_backtest
from indicators_multi_factor_v1 import compute_indicators
from metrics_multi_factor_v1 import compute_summary, save_summary
from plotting_multi_factor_v1 import plot_equity_comparison, plot_equity_curve
from strategy_versions_multi_factor_v1 import get_version_features


SUMMARY_COLUMNS = [
    "version",
    "total_return",
    "sharpe",
    "max_drawdown",
    "trade_count",
    "win_rate",
    "profit_factor",
    "avg_trade_pnl",
    "avg_win",
    "avg_loss",
    "long_trade_count",
    "short_trade_count",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_15m_data()
    validation = validate_15m_data(data)
    (OUTPUT_DIR / "data_validation_multi_factor_v1.json").write_text(pd.Series(validation).to_json(indent=2), encoding="utf-8")
    comparison_curves: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for version in VERSION_ORDER:
        features = get_version_features(version)
        version_dir = OUTPUT_DIR / version
        version_dir.mkdir(parents=True, exist_ok=True)
        indicator_data = compute_indicators(data, {"channel_mode": features.channel_mode})
        equity_df, trades_df = run_backtest(indicator_data.copy(), version)
        summary = compute_summary(equity_df, trades_df, INITIAL_CAPITAL)
        summary["version"] = version
        summary["channel_mode"] = features.channel_mode
        summary["data_start"] = validation["start"]
        summary["data_end"] = validation["end"]
        summary["data_bad_spacing_count"] = validation["bad_spacing_count"]
        equity_df.to_csv(version_dir / "equity_curve_multi_factor_v1.csv", index=False)
        daily_equity_df = equity_df.copy()
        daily_equity_df["date"] = pd.to_datetime(daily_equity_df["open_time"]).dt.date
        daily_equity_df = daily_equity_df.groupby("date", as_index=False).tail(1)
        daily_equity_df.to_csv(version_dir / "daily_equity_multi_factor_v1.csv", index=False)
        trades_df.to_csv(version_dir / "trades_multi_factor_v1.csv", index=False)
        save_summary(version_dir / "summary_multi_factor_v1.json", summary)
        subtitle = f"Return {summary['total_return']:.2%} | Sharpe {summary['sharpe']:.3f} | MaxDD {summary['max_drawdown']:.2%} | Trades {summary['trade_count']} | Channel {features.channel_mode}"
        plot_equity_curve(equity_df, version_dir / "equity_curve_multi_factor_v1.png", f"ASR BTC Multi Factor {version}", subtitle)
        comparison_curves[version] = equity_df
        summary_rows.append({col: summary.get(col) for col in SUMMARY_COLUMNS})

    summary_df = pd.DataFrame(summary_rows).sort_values(["sharpe", "total_return"], ascending=[False, False])
    summary_df.to_csv(OUTPUT_DIR / "summary_table_multi_factor_v1.csv", index=False)
    plot_equity_comparison(comparison_curves, OUTPUT_DIR / "equity_comparison_multi_factor_v1.png")


if __name__ == "__main__":
    main()
