from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config_channel_v4 import BASE_PARAMS, INITIAL_CAPITAL
from config_channel_v5 import OUTPUT_DIR as OUTPUT_DIR_V5
from data_loader_channel_v4 import load_15m_data, validate_15m_data
from engine_channel_v4 import (
    BacktestState,
    _close_all_positions,
    _close_position,
    _enter_position,
    _first_exit_hit,
    _update_mfe_mae,
    _value_or_none,
)
from indicators_channel_v4 import compute_indicators
from metrics_channel_v5 import compute_summary, save_summary
from plotting_channel_v5 import plot_equity_comparison, plot_equity_curve
from strategy_versions_channel_v4 import get_version_features


OUTPUT_DIR = OUTPUT_DIR_V5 / "v17_trend"
BASELINE_V17_DIR = Path(r"D:\workspace\20260325\asr_btc_channel_research_v1\output_channel_v4\v17")


def run_backtest_v17_trend_only(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = get_version_features("v17")
    params = dict(BASE_PARAMS)
    state = BacktestState(version="v17_trend")
    equity_rows: list[dict] = []

    required_cols = ["smoothRes", "smoothSup", "smoothMid", "midHigh", "midLow", "zoneOffset", "maLine", "rsiVal", "atr_", "dynamicOffset"]

    for _, row in data.iterrows():
        if any(pd.isna(row[col]) for col in required_cols):
            equity_rows.append({"open_time": row["open_time"], "equity": state.current_equity(float(row["close"])), "realized_equity": state.realized_equity})
            continue

        _update_mfe_mae(state, row)

        close = float(row["close"])
        is15m = True
        state.breakoutAboveBars = state.breakoutAboveBars + 1 if params["breakout_15m"] and is15m and close > float(row["superbuy"]) else 0
        state.breakoutBelowBars = state.breakoutBelowBars + 1 if params["breakout_15m"] and is15m and close < float(row["supersell"]) else 0
        breakout_ready = int(row["bar_index"]) - state.lastBreakoutBar >= params["breakout_cooldown_bars"]
        breakout_up = params["breakout_15m"] and is15m and state.trendMode == 0 and breakout_ready and state.breakoutAboveBars >= params["breakout_confirm_bars"]
        breakout_down = params["breakout_15m"] and is15m and state.trendMode == 0 and breakout_ready and state.breakoutBelowBars >= params["breakout_confirm_bars"]
        did_breakout_flip = False
        did_breakout_exit = False
        reverse_lock_active = state.trendMode in (2, -2) and (int(row["bar_index"]) - state.lastEntryBar) < features.reverse_lock_bars
        reverse_in_channel = close >= float(row["smoothSup"]) and close <= float(row["smoothRes"])

        avg_price = state.position_avg_price()
        if features.reverse_breakeven_on_channel and state.trendMode == 2 and state.is_long() and reverse_in_channel and avg_price is not None:
            base_val = avg_price if state.reverseStopPx is None else state.reverseStopPx
            state.reverseStopPx = max(base_val, avg_price)
        if features.reverse_breakeven_on_channel and state.trendMode == -2 and state.is_short() and reverse_in_channel and avg_price is not None:
            base_val = avg_price if state.reverseStopPx is None else state.reverseStopPx
            state.reverseStopPx = min(base_val, avg_price)

        if breakout_up and state.trendMode != 1:
            _close_all_positions(state, row, "Breakout Flip Up")
            qty = state.current_equity(close) / close
            state.nLong = 0
            state.nShort = 0
            state.entryPx1 = None
            state.entryPx2 = None
            state.entryPx3 = None
            state.be1 = False
            state.be2 = False
            state.be3 = False
            state.trendMode = 1
            state.breakoutInsideBars = 0
            state.breakoutEntryBar = int(row["bar_index"])
            state.lastBreakoutBar = int(row["bar_index"])
            state.reverseStopPx = None
            state.reverseTpPx = None
            state.lastEntryBar = int(row["bar_index"])
            _enter_position(state, "TrendLong", "long", qty, row, 0, breakout_context="breakout_up")
            did_breakout_flip = True

        if breakout_down and state.trendMode != -1:
            _close_all_positions(state, row, "Breakout Flip Down")
            qty = state.current_equity(close) / close
            state.nLong = 0
            state.nShort = 0
            state.entryPx1 = None
            state.entryPx2 = None
            state.entryPx3 = None
            state.be1 = False
            state.be2 = False
            state.be3 = False
            state.trendMode = -1
            state.breakoutInsideBars = 0
            state.breakoutEntryBar = int(row["bar_index"])
            state.lastBreakoutBar = int(row["bar_index"])
            state.reverseStopPx = None
            state.reverseTpPx = None
            state.lastEntryBar = int(row["bar_index"])
            _enter_position(state, "TrendShort", "short", qty, row, 0, breakout_context="breakout_down")
            did_breakout_flip = True

        if not did_breakout_flip and state.trendMode == 1:
            held_long_enough = state.breakoutEntryBar is not None and (int(row["bar_index"]) - state.breakoutEntryBar) >= params["breakout_min_hold_bars"]
            state.breakoutInsideBars = state.breakoutInsideBars + 1 if held_long_enough and close <= float(row["smoothRes"]) else 0
            breakout_long_tp = state.is_long() and float(row["rsiVal"]) >= params["breakout_rsi_high"]
            breakout_long_stop = state.breakoutInsideBars >= params["breakout_back_bars"]
            if breakout_long_tp:
                qty = state.current_equity(close) / close
                state.trendMode = -2
                state.breakoutInsideBars = 0
                state.breakoutEntryBar = None
                state.reverseStopPx = _value_or_none(row["reverseShortStopPxRef"])
                state.reverseTpPx = float(row["smoothMid"])
                state.lastBreakoutBar = int(row["bar_index"])
                state.lastEntryBar = int(row["bar_index"])
                _enter_position(state, "RevShort", "short", qty, row, 0, breakout_context="breakout_long_tp", reverse_context="rev_short")
                did_breakout_exit = True
            elif breakout_long_stop:
                _close_position(state, "TrendLong", row, close, "Breakout Stop", "TrendLongStop", "market")
                state.trendMode = 0
                state.breakoutInsideBars = 0
                state.breakoutEntryBar = None
                state.reverseStopPx = None
                state.reverseTpPx = None
                state.lastBreakoutBar = int(row["bar_index"])
                state.lastEntryBar = int(row["bar_index"])
                did_breakout_exit = True

        if not did_breakout_flip and state.trendMode == -1:
            held_long_enough = state.breakoutEntryBar is not None and (int(row["bar_index"]) - state.breakoutEntryBar) >= params["breakout_min_hold_bars"]
            state.breakoutInsideBars = state.breakoutInsideBars + 1 if held_long_enough and close >= float(row["smoothSup"]) else 0
            breakout_short_tp = state.is_short() and float(row["rsiVal"]) <= params["breakout_rsi_low"]
            breakout_short_stop = state.breakoutInsideBars >= params["breakout_back_bars"]
            if breakout_short_tp:
                qty = state.current_equity(close) / close
                state.trendMode = 2
                state.breakoutInsideBars = 0
                state.breakoutEntryBar = None
                state.reverseStopPx = _value_or_none(row["reverseLongStopPxRef"])
                state.reverseTpPx = float(row["smoothMid"])
                state.lastBreakoutBar = int(row["bar_index"])
                state.lastEntryBar = int(row["bar_index"])
                _enter_position(state, "RevLong", "long", qty, row, 0, breakout_context="breakout_short_tp", reverse_context="rev_long")
                did_breakout_exit = True
            elif breakout_short_stop:
                _close_position(state, "TrendShort", row, close, "Breakout Stop", "TrendShortStop", "market")
                state.trendMode = 0
                state.breakoutInsideBars = 0
                state.breakoutEntryBar = None
                state.reverseStopPx = None
                state.reverseTpPx = None
                state.lastBreakoutBar = int(row["bar_index"])
                state.lastEntryBar = int(row["bar_index"])
                did_breakout_exit = True

        if not did_breakout_flip and not did_breakout_exit and state.trendMode == 2 and state.reverseStopPx is not None and not reverse_lock_active and "RevLong" in state.open_positions:
            state.reverseTpPx = float(row["smoothMid"])
            hit = _first_exit_hit("long", state.reverseStopPx, state.reverseTpPx, row)
            if hit is not None:
                hit_kind, hit_price = hit
                _close_position(state, "RevLong", row, hit_price, "Reverse TP Channel" if hit_kind == "limit" else "Reverse Stop", "Exit RevLong", "limit" if hit_kind == "limit" else "stop")

        if not did_breakout_flip and not did_breakout_exit and state.trendMode == -2 and state.reverseStopPx is not None and not reverse_lock_active and "RevShort" in state.open_positions:
            state.reverseTpPx = float(row["smoothMid"])
            hit = _first_exit_hit("short", state.reverseStopPx, state.reverseTpPx, row)
            if hit is not None:
                hit_kind, hit_price = hit
                _close_position(state, "RevShort", row, hit_price, "Reverse TP Channel" if hit_kind == "limit" else "Reverse Stop", "Exit RevShort", "limit" if hit_kind == "limit" else "stop")

        if features.level0_mfe_zone_breakeven and not did_breakout_flip and not did_breakout_exit:
            zone_offset = float(row["zoneOffset"])
            for entry_id in ["TrendLong", "RevLong"]:
                pos = state.open_positions.get(entry_id)
                if pos is None:
                    continue
                bars_held = int(row["bar_index"]) - pos.entry_bar
                protect_price = pos.entry_price * (1.0 + params["be_buffer"])
                can_protect = entry_id == "TrendLong" and bars_held >= params["breakout_min_hold_bars"]
                can_protect = can_protect or (entry_id == "RevLong" and not reverse_lock_active)
                if can_protect and pos.mfe_abs >= zone_offset and close <= protect_price:
                    _close_position(state, entry_id, row, close, f"MFEProtect {entry_id}", f"MFEProtect {entry_id}", "market")
                    state.breakoutInsideBars = 0
                    state.breakoutEntryBar = None
                    state.reverseStopPx = None
                    state.reverseTpPx = None
                    state.lastBreakoutBar = int(row["bar_index"])
                    state.lastEntryBar = int(row["bar_index"])
                    if state.is_flat():
                        state.trendMode = 0
            for entry_id in ["TrendShort", "RevShort"]:
                pos = state.open_positions.get(entry_id)
                if pos is None:
                    continue
                bars_held = int(row["bar_index"]) - pos.entry_bar
                protect_price = pos.entry_price * (1.0 - params["be_buffer"])
                can_protect = entry_id == "TrendShort" and bars_held >= params["breakout_min_hold_bars"]
                can_protect = can_protect or (entry_id == "RevShort" and not reverse_lock_active)
                if can_protect and pos.mfe_abs >= zone_offset and close >= protect_price:
                    _close_position(state, entry_id, row, close, f"MFEProtect {entry_id}", f"MFEProtect {entry_id}", "market")
                    state.breakoutInsideBars = 0
                    state.breakoutEntryBar = None
                    state.reverseStopPx = None
                    state.reverseTpPx = None
                    state.lastBreakoutBar = int(row["bar_index"])
                    state.lastEntryBar = int(row["bar_index"])
                    if state.is_flat():
                        state.trendMode = 0

        if state.is_flat() and not did_breakout_flip:
            if state.trendMode in (2, -2):
                state.lastBreakoutBar = int(row["bar_index"])
                state.lastEntryBar = int(row["bar_index"])
            state.nLong = 0
            state.nShort = 0
            state.be1 = False
            state.be2 = False
            state.be3 = False
            state.entryPx1 = None
            state.entryPx2 = None
            state.entryPx3 = None
            state.breakoutInsideBars = 0
            state.breakoutEntryBar = None
            state.reverseStopPx = None
            state.reverseTpPx = None
            state.outsideLongBars = 0
            state.outsideShortBars = 0
            state.trendMode = 0

        equity_rows.append(
            {
                "open_time": row["open_time"],
                "equity": state.current_equity(close),
                "realized_equity": state.realized_equity,
            }
        )

    if state.open_positions:
        last_row = data.iloc[-1]
        for entry_id in list(state.open_positions.keys()):
            _close_position(state, entry_id, last_row, float(last_row["close"]), "Force Close End", "ForceClose", "market")
        equity_rows[-1]["equity"] = state.realized_equity
        equity_rows[-1]["realized_equity"] = state.realized_equity

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(state.trades)
    return equity_df, trades_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_15m_data()
    validation = validate_15m_data(data)
    features = get_version_features("v17")
    indicator_data = compute_indicators(data, {"channel_mode": features.channel_mode})
    equity_df, trades_df = run_backtest_v17_trend_only(indicator_data.copy())

    summary = compute_summary(equity_df, trades_df, INITIAL_CAPITAL)
    summary["version"] = "v17_trend"
    summary["source_version"] = "v17"
    summary["strategy_mode"] = "trend_only"
    summary["channel_mode"] = features.channel_mode
    summary["data_start"] = validation["start"]
    summary["data_end"] = validation["end"]
    summary["data_bad_spacing_count"] = validation["bad_spacing_count"]

    equity_df.to_csv(OUTPUT_DIR / "equity_curve_channel_v5.csv", index=False)
    daily_equity_df = equity_df.copy()
    daily_equity_df["date"] = pd.to_datetime(daily_equity_df["open_time"]).dt.date
    daily_equity_df = daily_equity_df.groupby("date", as_index=False).tail(1)
    daily_equity_df.to_csv(OUTPUT_DIR / "daily_equity_channel_v5.csv", index=False)
    trades_df.to_csv(OUTPUT_DIR / "trades_channel_v5.csv", index=False)
    save_summary(OUTPUT_DIR / "summary_channel_v5.json", summary)

    subtitle = f"Return {summary['total_return']:.2%} | Sharpe {summary['sharpe']:.3f} | MaxDD {summary['max_drawdown']:.2%} | Trades {summary['trade_count']} | Trend Only"
    plot_equity_curve(equity_df, OUTPUT_DIR / "equity_curve_channel_v5.png", "ASR BTC v17 Trend Only", subtitle)

    curves_with_trend: dict[str, pd.DataFrame] = {}
    for path in sorted(OUTPUT_DIR_V5.glob("v*/equity_curve_channel_v5.csv")):
        version = path.parent.name
        curves_with_trend[version] = pd.read_csv(path, parse_dates=["open_time"])
    curves_with_trend["v17_trend"] = equity_df.copy()
    plot_equity_comparison(curves_with_trend, OUTPUT_DIR_V5 / "equity_comparison_channel_v5_with_v17_trend.png")

    v17_full_curve = pd.read_csv(BASELINE_V17_DIR / "equity_curve_channel_v4.csv", parse_dates=["open_time"])
    plot_equity_comparison(
        {
            "v17_full": v17_full_curve,
            "v17_trend": equity_df.copy(),
        },
        OUTPUT_DIR / "equity_comparison_v17_full_vs_trend_only.png",
    )

    full_summary = json.loads((BASELINE_V17_DIR / "summary_channel_v4.json").read_text(encoding="utf-8"))
    comparison = {
        "v17_full_total_return": float(full_summary["total_return"]),
        "v17_trend_total_return": float(summary["total_return"]),
        "v17_full_sharpe": float(full_summary["sharpe"]),
        "v17_trend_sharpe": float(summary["sharpe"]),
        "v17_full_max_drawdown": float(full_summary["max_drawdown"]),
        "v17_trend_max_drawdown": float(summary["max_drawdown"]),
        "v17_full_trade_count": int(full_summary["trade_count"]),
        "v17_trend_trade_count": int(summary["trade_count"]),
        "trend_to_full_profit_ratio": float(summary["total_return"] / full_summary["total_return"]) if float(full_summary["total_return"]) != 0 else None,
    }
    (OUTPUT_DIR / "comparison_vs_v17_full.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
