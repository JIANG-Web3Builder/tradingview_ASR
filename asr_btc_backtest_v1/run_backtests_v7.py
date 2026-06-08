from __future__ import annotations

import pandas as pd

from config_v7 import INITIAL_CAPITAL, OUTPUT_DIR, VERSION_ORDER
from data_loader_v1 import load_15m_data, validate_15m_data
from engine_v7 import run_backtest
from indicators_v1 import compute_indicators
from metrics_v1 import compute_summary, save_summary
from plotting_v1 import plot_equity_comparison, plot_equity_curve


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_15m_data()
    validation = validate_15m_data(data)
    (OUTPUT_DIR / "data_validation_v7.json").write_text(pd.Series(validation).to_json(indent=2), encoding="utf-8")
    indicator_data = compute_indicators(data)
    comparison_curves: dict[str, pd.DataFrame] = {}

    for version in VERSION_ORDER:
        version_dir = OUTPUT_DIR / version
        version_dir.mkdir(parents=True, exist_ok=True)
        equity_df, trades_df = run_backtest(indicator_data.copy(), version)
        summary = compute_summary(equity_df, trades_df, INITIAL_CAPITAL)
        summary["version"] = version
        summary["data_start"] = validation["start"]
        summary["data_end"] = validation["end"]
        summary["data_bad_spacing_count"] = validation["bad_spacing_count"]
        equity_df.to_csv(version_dir / "equity_curve_v7.csv", index=False)
        daily_equity_df = equity_df.copy()
        daily_equity_df["date"] = pd.to_datetime(daily_equity_df["open_time"]).dt.date
        daily_equity_df = daily_equity_df.groupby("date", as_index=False).tail(1)
        daily_equity_df.to_csv(version_dir / "daily_equity_v7.csv", index=False)
        trades_df.to_csv(version_dir / "trades_v7.csv", index=False)
        save_summary(version_dir / "summary_v7.json", summary)
        subtitle = f"Return {summary['total_return']:.2%} | Sharpe {summary['sharpe']:.3f} | MaxDD {summary['max_drawdown']:.2%} | Trades {summary['trade_count']}"
        plot_equity_curve(equity_df, version_dir / "equity_curve_v7.png", f"ASR BTC {version}", subtitle)
        comparison_curves[version] = equity_df

    plot_equity_comparison(comparison_curves, OUTPUT_DIR / "equity_comparison_v7.png")


if __name__ == "__main__":
    main()
