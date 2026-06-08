from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\workspace\20260325\asr_btc_channel_research_v1")
SOURCE_DIR = BASE_DIR / "output_channel_v4" / "v17"
OUTPUT_DIR = BASE_DIR / "v17_entry_path_analysis_v1"
DATA_FILE = Path(r"D:\workspace\20260325\data\BTCUSDT_15m.csv")
TRADES_FILE = SOURCE_DIR / "trades_channel_v4.csv"
EQUITY_FILE = SOURCE_DIR / "equity_curve_channel_v4.csv"
SUMMARY_FILE = SOURCE_DIR / "summary_channel_v4.json"

START_DATE = pd.Timestamp("2024-01-01 00:00:00")
END_DATE = pd.Timestamp("2026-05-01 23:59:59")
TIMEFRAME_MINUTES = 15
HORIZONS = [1, 2, 4, 8, 12, 24, 48, 96, 192]
ADVERSE_THRESHOLDS_PCT = [0.25, 0.5, 1.0, 2.0, 3.0]
FAVORABLE_THRESHOLDS_PCT = [0.25, 0.5, 1.0, 2.0, 3.0]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    prices = pd.read_csv(DATA_FILE)
    prices["open_time"] = pd.to_datetime(prices["open_time"], utc=False)
    prices = prices[(prices["open_time"] >= START_DATE) & (prices["open_time"] <= END_DATE)].copy()
    prices = prices.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    prices["bar_index"] = np.arange(len(prices))
    for col in ["open", "high", "low", "close", "volume"]:
        prices[col] = prices[col].astype(float)

    trades = pd.read_csv(TRADES_FILE)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=False)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=False)
    numeric_cols = ["trade_id", "entry_price", "exit_price", "qty", "bars_held", "pnl_abs", "pnl_pct", "mfe_abs", "mae_abs", "mfe_pct", "mae_pct", "level_snapshot", "trend_mode_at_entry"]
    for col in numeric_cols:
        trades[col] = pd.to_numeric(trades[col], errors="coerce")

    equity = pd.read_csv(EQUITY_FILE)
    equity["open_time"] = pd.to_datetime(equity["open_time"], utc=False)
    for col in ["equity", "realized_equity"]:
        equity[col] = equity[col].astype(float)

    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    return prices, trades, equity, summary


def side_sign(side: str) -> int:
    return 1 if side == "long" else -1


def signed_return_pct(price: float, entry_price: float, side: str) -> float:
    return (price / entry_price - 1.0) * 100.0 * side_sign(side)


def calc_underwater_stats(path: pd.DataFrame, entry_price: float, side: str) -> dict[str, float | int]:
    sign = side_sign(side)
    close_ret = (path["close"].to_numpy(dtype=float) / entry_price - 1.0) * 100.0 * sign
    adverse_abs = np.where(sign == 1, np.maximum(0.0, entry_price - path["low"].to_numpy(dtype=float)), np.maximum(0.0, path["high"].to_numpy(dtype=float) - entry_price))
    adverse_pct = adverse_abs / entry_price * 100.0
    is_underwater = close_ret < 0.0
    underwater_bars = int(is_underwater.sum())
    max_consecutive_underwater_bars = 0
    current = 0
    for flag in is_underwater:
        if flag:
            current += 1
            max_consecutive_underwater_bars = max(max_consecutive_underwater_bars, current)
        else:
            current = 0
    avg_underwater_close_pct = float(close_ret[is_underwater].mean()) if underwater_bars else 0.0
    return {
        "underwater_bars": underwater_bars,
        "underwater_minutes": underwater_bars * TIMEFRAME_MINUTES,
        "underwater_ratio": underwater_bars / len(path) if len(path) else 0.0,
        "max_consecutive_underwater_bars": max_consecutive_underwater_bars,
        "max_consecutive_underwater_minutes": max_consecutive_underwater_bars * TIMEFRAME_MINUTES,
        "avg_underwater_close_pct": avg_underwater_close_pct,
        "max_adverse_pct_path": float(adverse_pct.max()) if len(adverse_pct) else 0.0,
    }


def first_threshold_bar(values: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(values >= threshold)
    if len(hits) == 0:
        return None
    return int(hits[0])


def observed_minutes(hit_bar: int | None) -> int | None:
    return None if hit_bar is None else (hit_bar + 1) * TIMEFRAME_MINUTES


def build_trade_path_rows(prices: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_by_time = {ts: idx for idx, ts in enumerate(prices["open_time"])}
    rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []

    for trade in trades.itertuples(index=False):
        entry_idx = price_by_time.get(trade.entry_time)
        exit_idx = price_by_time.get(trade.exit_time)
        if entry_idx is None or exit_idx is None:
            continue
        if exit_idx < entry_idx:
            continue

        observation_start_idx = entry_idx + 1
        path = prices.iloc[observation_start_idx : exit_idx + 1].copy()
        entry_price = float(trade.entry_price)
        side = str(trade.side)
        sign = side_sign(side)
        highs = path["high"].to_numpy(dtype=float)
        lows = path["low"].to_numpy(dtype=float)
        closes = path["close"].to_numpy(dtype=float)
        favorable_abs = np.where(sign == 1, np.maximum(0.0, highs - entry_price), np.maximum(0.0, entry_price - lows))
        adverse_abs = np.where(sign == 1, np.maximum(0.0, entry_price - lows), np.maximum(0.0, highs - entry_price))
        favorable_pct = favorable_abs / entry_price * 100.0
        adverse_pct = adverse_abs / entry_price * 100.0
        close_ret_pct = (closes / entry_price - 1.0) * 100.0 * sign

        first_adverse_bar = first_threshold_bar(adverse_abs, 0.01)
        first_favorable_bar = first_threshold_bar(favorable_abs, 0.01)
        first_material_adverse_bar = first_threshold_bar(adverse_pct, 0.5)
        first_material_favorable_bar = first_threshold_bar(favorable_pct, 0.5)
        underwater = calc_underwater_stats(path, entry_price, side)
        max_adverse_idx = int(np.argmax(adverse_pct)) if len(adverse_pct) else 0
        max_favorable_idx = int(np.argmax(favorable_pct)) if len(favorable_pct) else 0
        minutes_to_max_adverse = (max_adverse_idx + 1) * TIMEFRAME_MINUTES if len(adverse_pct) else None
        minutes_to_max_favorable = (max_favorable_idx + 1) * TIMEFRAME_MINUTES if len(favorable_pct) else None
        ended_profit = float(trade.pnl_abs) > 0.0
        ended_loss = float(trade.pnl_abs) < 0.0
        first_move = "none"
        if first_adverse_bar is not None and first_favorable_bar is not None:
            first_move = "adverse" if first_adverse_bar < first_favorable_bar else "favorable" if first_favorable_bar < first_adverse_bar else "same_bar"
        elif first_adverse_bar is not None:
            first_move = "adverse"
        elif first_favorable_bar is not None:
            first_move = "favorable"

        base = {
            "trade_id": int(trade.trade_id),
            "side": side,
            "entry_id": str(trade.entry_id),
            "level_snapshot": int(trade.level_snapshot) if not pd.isna(trade.level_snapshot) else -1,
            "trend_mode_at_entry": int(trade.trend_mode_at_entry) if not pd.isna(trade.trend_mode_at_entry) else 0,
            "exit_reason": str(trade.exit_reason),
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "entry_price": entry_price,
            "exit_price": float(trade.exit_price),
            "bars_held": int(trade.bars_held),
            "hold_minutes": int(trade.bars_held) * TIMEFRAME_MINUTES,
            "pnl_abs": float(trade.pnl_abs),
            "pnl_pct": float(trade.pnl_pct),
            "ended_profit": ended_profit,
            "ended_loss": ended_loss,
            "mfe_abs_trade": float(trade.mfe_abs),
            "mae_abs_trade": float(trade.mae_abs),
            "mfe_pct_trade": float(trade.mfe_pct),
            "mae_pct_trade": float(trade.mae_pct),
            "max_favorable_pct_path": float(favorable_pct.max()) if len(favorable_pct) else 0.0,
            "max_adverse_pct_path": float(adverse_pct.max()) if len(adverse_pct) else 0.0,
            "bar_of_max_favorable": max_favorable_idx,
            "bar_of_max_adverse": max_adverse_idx,
            "minutes_to_max_favorable": minutes_to_max_favorable,
            "minutes_to_max_adverse": minutes_to_max_adverse,
            "first_move": first_move,
            "first_adverse_bar": first_adverse_bar,
            "first_favorable_bar": first_favorable_bar,
            "first_material_adverse_bar": first_material_adverse_bar,
            "first_material_favorable_bar": first_material_favorable_bar,
            "first_adverse_minutes": observed_minutes(first_adverse_bar),
            "first_favorable_minutes": observed_minutes(first_favorable_bar),
            "first_material_adverse_minutes": observed_minutes(first_material_adverse_bar),
            "first_material_favorable_minutes": observed_minutes(first_material_favorable_bar),
        }
        base.update(underwater)
        rows.append(base)

        for h in HORIZONS:
            horizon_idx = entry_idx + h
            available = horizon_idx <= exit_idx and horizon_idx < len(prices)
            if available:
                window = prices.iloc[observation_start_idx : horizon_idx + 1]
                close_at_h = float(prices.iloc[horizon_idx]["close"])
                window_highs = window["high"].to_numpy(dtype=float)
                window_lows = window["low"].to_numpy(dtype=float)
                window_favorable = np.where(sign == 1, np.maximum(0.0, window_highs - entry_price), np.maximum(0.0, entry_price - window_lows)) / entry_price * 100.0
                window_adverse = np.where(sign == 1, np.maximum(0.0, entry_price - window_lows), np.maximum(0.0, window_highs - entry_price)) / entry_price * 100.0
                close_ret_h = signed_return_pct(close_at_h, entry_price, side)
                min_close_ret_h = float(((window["close"].to_numpy(dtype=float) / entry_price - 1.0) * 100.0 * sign).min())
                max_close_ret_h = float(((window["close"].to_numpy(dtype=float) / entry_price - 1.0) * 100.0 * sign).max())
            else:
                close_at_h = np.nan
                close_ret_h = np.nan
                min_close_ret_h = np.nan
                max_close_ret_h = np.nan
                window_favorable = np.array([], dtype=float)
                window_adverse = np.array([], dtype=float)
            horizon_rows.append({
                "trade_id": int(trade.trade_id),
                "side": side,
                "entry_id": str(trade.entry_id),
                "level_snapshot": int(trade.level_snapshot) if not pd.isna(trade.level_snapshot) else -1,
                "trend_mode_at_entry": int(trade.trend_mode_at_entry) if not pd.isna(trade.trend_mode_at_entry) else 0,
                "exit_reason": str(trade.exit_reason),
                "horizon_bars": h,
                "horizon_minutes": h * TIMEFRAME_MINUTES,
                "available": available,
                "close_at_h": close_at_h,
                "close_ret_pct_at_h": close_ret_h,
                "profitable_at_h": bool(close_ret_h > 0.0) if available else np.nan,
                "adverse_at_h": bool(close_ret_h < 0.0) if available else np.nan,
                "flat_at_h": bool(abs(close_ret_h) < 1e-12) if available else np.nan,
                "max_favorable_pct_to_h": float(window_favorable.max()) if available else np.nan,
                "max_adverse_pct_to_h": float(window_adverse.max()) if available else np.nan,
                "min_close_ret_pct_to_h": min_close_ret_h,
                "max_close_ret_pct_to_h": max_close_ret_h,
                "ended_profit": ended_profit,
                "ended_loss": ended_loss,
                "pnl_abs": float(trade.pnl_abs),
                "pnl_pct": float(trade.pnl_pct),
            })

        for threshold in ADVERSE_THRESHOLDS_PCT:
            hit_bar = first_threshold_bar(adverse_pct, threshold)
            threshold_rows.append({
                "trade_id": int(trade.trade_id),
                "side": side,
                "entry_id": str(trade.entry_id),
                "level_snapshot": int(trade.level_snapshot) if not pd.isna(trade.level_snapshot) else -1,
                "threshold_type": "adverse_pct",
                "threshold_pct": threshold,
                "hit": hit_bar is not None,
                "hit_bar": hit_bar,
                "hit_minutes": observed_minutes(hit_bar),
                "ended_profit": ended_profit,
                "ended_loss": ended_loss,
            })
        for threshold in FAVORABLE_THRESHOLDS_PCT:
            hit_bar = first_threshold_bar(favorable_pct, threshold)
            threshold_rows.append({
                "trade_id": int(trade.trade_id),
                "side": side,
                "entry_id": str(trade.entry_id),
                "level_snapshot": int(trade.level_snapshot) if not pd.isna(trade.level_snapshot) else -1,
                "threshold_type": "favorable_pct",
                "threshold_pct": threshold,
                "hit": hit_bar is not None,
                "hit_bar": hit_bar,
                "hit_minutes": observed_minutes(hit_bar),
                "ended_profit": ended_profit,
                "ended_loss": ended_loss,
            })

    return pd.DataFrame(rows), pd.DataFrame(horizon_rows), pd.DataFrame(threshold_rows)


def summarize_bool_rate(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        valid = group[value_col].dropna()
        row = {col: key for col, key in zip(group_cols, keys)}
        row["sample_count"] = int(len(valid))
        row[f"{value_col}_rate"] = float(valid.astype(bool).mean()) if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def agg_trade_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "trade_count": int(len(group)),
            "win_rate": float(group["ended_profit"].mean()) if len(group) else np.nan,
            "loss_rate": float(group["ended_loss"].mean()) if len(group) else np.nan,
            "avg_pnl_abs": float(group["pnl_abs"].mean()),
            "median_pnl_abs": float(group["pnl_abs"].median()),
            "avg_pnl_pct": float(group["pnl_pct"].mean()),
            "median_pnl_pct": float(group["pnl_pct"].median()),
            "avg_bars_held": float(group["bars_held"].mean()),
            "median_bars_held": float(group["bars_held"].median()),
            "avg_hold_hours": float(group["hold_minutes"].mean() / 60.0),
            "avg_underwater_hours": float(group["underwater_minutes"].mean() / 60.0),
            "median_underwater_hours": float(group["underwater_minutes"].median() / 60.0),
            "avg_underwater_ratio": float(group["underwater_ratio"].mean()),
            "avg_max_consecutive_underwater_hours": float(group["max_consecutive_underwater_minutes"].mean() / 60.0),
            "avg_max_adverse_pct": float(group["max_adverse_pct_path"].mean()),
            "median_max_adverse_pct": float(group["max_adverse_pct_path"].median()),
            "avg_max_favorable_pct": float(group["max_favorable_pct_path"].mean()),
            "median_max_favorable_pct": float(group["max_favorable_pct_path"].median()),
            "first_move_adverse_rate": float((group["first_move"] == "adverse").mean()),
            "first_move_favorable_rate": float((group["first_move"] == "favorable").mean()),
            "material_adverse_0p5_rate": float(group["first_material_adverse_bar"].notna().mean()),
            "material_favorable_0p5_rate": float(group["first_material_favorable_bar"].notna().mean()),
            "avg_minutes_to_max_adverse": float(group["minutes_to_max_adverse"].mean()),
            "avg_minutes_to_max_favorable": float(group["minutes_to_max_favorable"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_table(df: pd.DataFrame, columns: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = []
    rows = []
    groups = [((), df)] if not group_cols else list(df.groupby(group_cols, dropna=False))
    for keys, group in groups:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        for col in columns:
            values = group[col].dropna()
            row = {group_col: key for group_col, key in zip(group_cols, keys)}
            row["metric"] = col
            row["count"] = int(len(values))
            if len(values):
                for q in QUANTILES:
                    row[f"q{int(q * 100):02d}"] = float(values.quantile(q))
                row["mean"] = float(values.mean())
            rows.append(row)
    return pd.DataFrame(rows)


def compute_equity_stats(equity: pd.DataFrame) -> pd.DataFrame:
    e = equity.copy().sort_values("open_time")
    e["ret"] = e["equity"].pct_change().fillna(0.0)
    e["roll_max"] = e["equity"].cummax()
    e["drawdown"] = e["equity"] / e["roll_max"] - 1.0
    e["month"] = e["open_time"].dt.to_period("M").astype(str)
    monthly = e.groupby("month").agg(
        start_equity=("equity", "first"),
        end_equity=("equity", "last"),
        min_drawdown=("drawdown", "min"),
        realized_end=("realized_equity", "last"),
    ).reset_index()
    monthly["return_pct"] = (monthly["end_equity"] / monthly["start_equity"] - 1.0) * 100.0
    return monthly


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value * 100.0:.2f}%"


def report_text(summary: dict[str, Any], trade_stats: pd.DataFrame, horizon_summary: pd.DataFrame, loss_stats: pd.DataFrame, threshold_summary: pd.DataFrame) -> str:
    overall = trade_stats.iloc[0]
    lines = []
    lines.append("# v17 Entry Path Risk Analysis v1")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append(f"- Version: {summary.get('version', 'v17')}")
    lines.append(f"- Channel mode: {summary.get('channel_mode', 'baseline')}")
    lines.append(f"- Data: {summary.get('data_start')} to {summary.get('data_end')}")
    lines.append(f"- Source trades: `{TRADES_FILE}`")
    lines.append("- Path observation starts from the next 15m bar after the entry close")
    lines.append("")
    lines.append("## Core Backtest Summary")
    lines.append("")
    lines.append(f"- Total return: {format_pct(float(summary['total_return']))}")
    lines.append(f"- Sharpe: {float(summary['sharpe']):.3f}")
    lines.append(f"- Max drawdown: {format_pct(float(summary['max_drawdown']))}")
    lines.append(f"- Trade count: {int(summary['trade_count'])}")
    lines.append(f"- Win rate: {format_pct(float(summary['win_rate']))}")
    lines.append(f"- Profit factor: {float(summary['profit_factor']):.3f}")
    lines.append("")
    lines.append("## Entry Path Summary")
    lines.append("")
    lines.append(f"- First move adverse rate: {overall['first_move_adverse_rate'] * 100.0:.2f}%")
    lines.append(f"- First move favorable rate: {overall['first_move_favorable_rate'] * 100.0:.2f}%")
    lines.append(f"- Hit adverse 0.5% sometime during trade: {overall['material_adverse_0p5_rate'] * 100.0:.2f}%")
    lines.append(f"- Hit favorable 0.5% sometime during trade: {overall['material_favorable_0p5_rate'] * 100.0:.2f}%")
    lines.append(f"- Avg max adverse excursion: {overall['avg_max_adverse_pct']:.2f}%")
    lines.append(f"- Median max adverse excursion: {overall['median_max_adverse_pct']:.2f}%")
    lines.append(f"- Avg underwater time: {overall['avg_underwater_hours']:.2f} hours")
    lines.append(f"- Avg underwater ratio while holding: {overall['avg_underwater_ratio'] * 100.0:.2f}%")
    lines.append(f"- Avg max consecutive underwater time: {overall['avg_max_consecutive_underwater_hours']:.2f} hours")
    lines.append("")
    lines.append("## Profit Accuracy After Entry Horizons")
    lines.append("")
    lines.append("| Horizon | Samples | Profitable rate | Adverse-close rate | Avg close return | Avg max adverse to horizon |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in horizon_summary.sort_values("horizon_bars").itertuples(index=False):
        lines.append(f"| {int(row.horizon_minutes)} min | {int(row.sample_count)} | {row.profitable_rate * 100.0:.2f}% | {row.adverse_rate * 100.0:.2f}% | {row.avg_close_ret_pct_at_h:.3f}% | {row.avg_max_adverse_pct_to_h:.3f}% |")
    lines.append("")
    lines.append("## Losing Trade Carry Summary")
    lines.append("")
    if len(loss_stats):
        loss = loss_stats.iloc[0]
        lines.append(f"- Losing trades: {int(loss['trade_count'])}")
        lines.append(f"- Avg losing hold time: {loss['avg_hold_hours']:.2f} hours")
        lines.append(f"- Median losing hold time: {loss['median_hold_hours']:.2f} hours")
        lines.append(f"- Avg underwater time in losing trades: {loss['avg_underwater_hours']:.2f} hours")
        lines.append(f"- Median underwater time in losing trades: {loss['median_underwater_hours']:.2f} hours")
        lines.append(f"- Avg max adverse excursion in losing trades: {loss['avg_max_adverse_pct']:.2f}%")
        lines.append(f"- Median max adverse excursion in losing trades: {loss['median_max_adverse_pct']:.2f}%")
        lines.append(f"- Avg max consecutive underwater time in losing trades: {loss['avg_max_consecutive_underwater_hours']:.2f} hours")
    lines.append("")
    lines.append("## Adverse/Favorable Threshold Hit Rates")
    lines.append("")
    lines.append("| Type | Threshold | Hit rate | Avg hit minutes | Samples |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in threshold_summary.sort_values(["threshold_type", "threshold_pct"]).itertuples(index=False):
        lines.append(f"| {row.threshold_type} | {row.threshold_pct:.2f}% | {row.hit_rate * 100.0:.2f}% | {row.avg_hit_minutes:.1f} | {int(row.sample_count)} |")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `trade_path_metrics_v1.csv`")
    lines.append("- `horizon_trade_metrics_v1.csv`")
    lines.append("- `horizon_summary_v1.csv`")
    lines.append("- `threshold_summary_v1.csv`")
    lines.append("- `trade_group_summary_v1.csv`")
    lines.append("- `loss_carry_summary_v1.csv`")
    lines.append("- `metric_quantiles_v1.csv`")
    lines.append("- `monthly_equity_summary_v1.csv`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices, trades, equity, summary = load_inputs()
    trade_paths, horizon_metrics, threshold_metrics = build_trade_path_rows(prices, trades)

    horizon_available = horizon_metrics[horizon_metrics["available"]].copy()
    horizon_summary = horizon_available.groupby(["horizon_bars", "horizon_minutes"], as_index=False).agg(
        sample_count=("trade_id", "count"),
        profitable_rate=("profitable_at_h", "mean"),
        adverse_rate=("adverse_at_h", "mean"),
        avg_close_ret_pct_at_h=("close_ret_pct_at_h", "mean"),
        median_close_ret_pct_at_h=("close_ret_pct_at_h", "median"),
        avg_max_favorable_pct_to_h=("max_favorable_pct_to_h", "mean"),
        avg_max_adverse_pct_to_h=("max_adverse_pct_to_h", "mean"),
    )

    horizon_by_side = horizon_available.groupby(["side", "horizon_bars", "horizon_minutes"], as_index=False).agg(
        sample_count=("trade_id", "count"),
        profitable_rate=("profitable_at_h", "mean"),
        adverse_rate=("adverse_at_h", "mean"),
        avg_close_ret_pct_at_h=("close_ret_pct_at_h", "mean"),
        avg_max_favorable_pct_to_h=("max_favorable_pct_to_h", "mean"),
        avg_max_adverse_pct_to_h=("max_adverse_pct_to_h", "mean"),
    )

    threshold_summary_rows = []
    for keys, group in threshold_metrics.groupby(["threshold_type", "threshold_pct"], dropna=False):
        hit_group = group[group["hit"]]
        threshold_summary_rows.append({
            "threshold_type": keys[0],
            "threshold_pct": keys[1],
            "sample_count": int(len(group)),
            "hit_count": int(len(hit_group)),
            "hit_rate": float(group["hit"].mean()) if len(group) else np.nan,
            "avg_hit_minutes": float(hit_group["hit_minutes"].mean()) if len(hit_group) else np.nan,
            "median_hit_minutes": float(hit_group["hit_minutes"].median()) if len(hit_group) else np.nan,
            "win_rate_after_hit": float(hit_group["ended_profit"].mean()) if len(hit_group) else np.nan,
            "loss_rate_after_hit": float(hit_group["ended_loss"].mean()) if len(hit_group) else np.nan,
        })
    threshold_summary = pd.DataFrame(threshold_summary_rows)

    trade_stats_overall = agg_trade_stats(trade_paths.assign(group="overall"), ["group"])
    group_summaries = [
        trade_stats_overall,
        agg_trade_stats(trade_paths, ["side"]),
        agg_trade_stats(trade_paths, ["entry_id"]),
        agg_trade_stats(trade_paths, ["level_snapshot"]),
        agg_trade_stats(trade_paths, ["side", "level_snapshot"]),
        agg_trade_stats(trade_paths, ["exit_reason"]),
    ]
    trade_group_summary = pd.concat(group_summaries, ignore_index=True, sort=False)

    losing = trade_paths[trade_paths["ended_loss"]].copy()
    loss_carry_summary = losing.assign(group="losing_trades").groupby("group", as_index=False).agg(
        trade_count=("trade_id", "count"),
        avg_hold_hours=("hold_minutes", lambda s: float(s.mean() / 60.0)),
        median_hold_hours=("hold_minutes", lambda s: float(s.median() / 60.0)),
        avg_underwater_hours=("underwater_minutes", lambda s: float(s.mean() / 60.0)),
        median_underwater_hours=("underwater_minutes", lambda s: float(s.median() / 60.0)),
        avg_underwater_ratio=("underwater_ratio", "mean"),
        avg_max_consecutive_underwater_hours=("max_consecutive_underwater_minutes", lambda s: float(s.mean() / 60.0)),
        avg_max_adverse_pct=("max_adverse_pct_path", "mean"),
        median_max_adverse_pct=("max_adverse_pct_path", "median"),
        avg_max_favorable_pct=("max_favorable_pct_path", "mean"),
        median_pnl_abs=("pnl_abs", "median"),
        avg_pnl_abs=("pnl_abs", "mean"),
    )

    loss_carry_by_entry = losing.groupby(["side", "entry_id"], as_index=False).agg(
        trade_count=("trade_id", "count"),
        avg_hold_hours=("hold_minutes", lambda s: float(s.mean() / 60.0)),
        median_hold_hours=("hold_minutes", lambda s: float(s.median() / 60.0)),
        avg_underwater_hours=("underwater_minutes", lambda s: float(s.mean() / 60.0)),
        median_underwater_hours=("underwater_minutes", lambda s: float(s.median() / 60.0)),
        avg_max_adverse_pct=("max_adverse_pct_path", "mean"),
        median_max_adverse_pct=("max_adverse_pct_path", "median"),
        avg_max_consecutive_underwater_hours=("max_consecutive_underwater_minutes", lambda s: float(s.mean() / 60.0)),
        avg_pnl_abs=("pnl_abs", "mean"),
    )

    metric_quantiles = quantile_table(
        trade_paths,
        ["pnl_abs", "pnl_pct", "bars_held", "underwater_minutes", "underwater_ratio", "max_adverse_pct_path", "max_favorable_pct_path", "minutes_to_max_adverse", "minutes_to_max_favorable"],
    )
    metric_quantiles_by_side = quantile_table(
        trade_paths,
        ["pnl_abs", "pnl_pct", "bars_held", "underwater_minutes", "max_adverse_pct_path", "max_favorable_pct_path"],
        ["side"],
    )
    monthly_equity = compute_equity_stats(equity)

    trade_paths.to_csv(OUTPUT_DIR / "trade_path_metrics_v1.csv", index=False)
    horizon_metrics.to_csv(OUTPUT_DIR / "horizon_trade_metrics_v1.csv", index=False)
    horizon_summary.to_csv(OUTPUT_DIR / "horizon_summary_v1.csv", index=False)
    horizon_by_side.to_csv(OUTPUT_DIR / "horizon_by_side_v1.csv", index=False)
    threshold_metrics.to_csv(OUTPUT_DIR / "threshold_trade_metrics_v1.csv", index=False)
    threshold_summary.to_csv(OUTPUT_DIR / "threshold_summary_v1.csv", index=False)
    trade_group_summary.to_csv(OUTPUT_DIR / "trade_group_summary_v1.csv", index=False)
    loss_carry_summary.to_csv(OUTPUT_DIR / "loss_carry_summary_v1.csv", index=False)
    loss_carry_by_entry.to_csv(OUTPUT_DIR / "loss_carry_by_entry_v1.csv", index=False)
    metric_quantiles.to_csv(OUTPUT_DIR / "metric_quantiles_v1.csv", index=False)
    metric_quantiles_by_side.to_csv(OUTPUT_DIR / "metric_quantiles_by_side_v1.csv", index=False)
    monthly_equity.to_csv(OUTPUT_DIR / "monthly_equity_summary_v1.csv", index=False)

    report = report_text(summary, trade_stats_overall, horizon_summary, loss_carry_summary, threshold_summary)
    (OUTPUT_DIR / "v17_entry_path_analysis_report_v1.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
