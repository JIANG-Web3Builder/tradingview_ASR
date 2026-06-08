from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd

from config_polymarket_multi_horizon_v1 import FEE_DRAGS, FIXED_STAKE, HORIZON_BARS, INITIAL_BANKROLL, TIMEFRAME_MINUTES, TIE_MODE


def side_sign(side: str) -> int:
    return 1 if side == "long" else -1


def build_binary_bets(signals: pd.DataFrame, prices: pd.DataFrame, horizons: Iterable[int] | None = None) -> pd.DataFrame:
    horizon_values = list(horizons or HORIZON_BARS)
    price_by_index = prices.set_index("bar_index")
    rows: list[dict] = []

    for signal in signals.itertuples(index=False):
        entry_idx = int(signal.signal_bar_index)
        entry_close = float(signal.signal_close)
        sign = side_sign(str(signal.side))
        for horizon_bars in horizon_values:
            settlement_idx = entry_idx + int(horizon_bars)
            if settlement_idx not in price_by_index.index:
                continue
            settle = price_by_index.loc[settlement_idx]
            settle_close = float(settle["close"])
            direction_return_pct = (settle_close / entry_close - 1.0) * 100.0 * sign
            if direction_return_pct > 0:
                won = True
                tie = False
            elif direction_return_pct < 0:
                won = False
                tie = False
            else:
                tie = True
                won = False if TIE_MODE == "loss" else np.nan
            rows.append({
                "signal_id": int(signal.signal_id),
                "trade_id": int(signal.trade_id),
                "version": str(signal.version),
                "side": str(signal.side),
                "entry_id": str(signal.entry_id),
                "level_snapshot": int(signal.level_snapshot) if not pd.isna(signal.level_snapshot) else -1,
                "trend_mode_at_entry": int(signal.trend_mode_at_entry) if not pd.isna(signal.trend_mode_at_entry) else 0,
                "entry_time": signal.entry_time,
                "settlement_time": settle["open_time"],
                "signal_bar_index": entry_idx,
                "settlement_bar_index": settlement_idx,
                "horizon_bars": int(horizon_bars),
                "horizon_minutes": int(horizon_bars) * TIMEFRAME_MINUTES,
                "signal_close": entry_close,
                "settlement_close": settle_close,
                "direction_return_pct": direction_return_pct,
                "won": won,
                "tie": tie,
                "signal_hour": int(signal.signal_hour),
                "signal_month": str(signal.signal_month),
                "candle_direction": str(signal.candle_direction),
                "candle_body_pct": float(signal.candle_body_pct),
                "rsiVal": float(signal.rsiVal),
                "rsi_bucket": str(signal.rsi_bucket),
                "atrPct": float(signal.atrPct),
                "atr_pct_bucket": str(signal.atr_pct_bucket),
                "zoneOffset": float(signal.zoneOffset),
                "channelWidth": float(signal.channelWidth),
                "breakout_context": str(signal.breakout_context) if not pd.isna(signal.breakout_context) else "",
                "reverse_context": str(signal.reverse_context) if not pd.isna(signal.reverse_context) else "",
            })

    return pd.DataFrame(rows)


def build_concurrency_scenarios(binary_bets: pd.DataFrame) -> pd.DataFrame:
    if binary_bets.empty:
        return binary_bets.copy()

    ordered = binary_bets.sort_values(["horizon_bars", "settlement_time", "entry_time", "signal_id"]).copy()
    independent = ordered.copy()
    independent["concurrency_mode"] = "independent"
    independent["source_signal_count"] = 1

    first_counts = ordered.groupby(["horizon_bars", "settlement_time"], dropna=False)["signal_id"].transform("count")
    first_signal = ordered.copy()
    first_signal["source_signal_count"] = first_counts
    first_signal = first_signal.groupby(["horizon_bars", "settlement_time"], as_index=False, dropna=False).head(1).copy()
    first_signal["concurrency_mode"] = "first_signal"

    same_direction_counts = ordered.groupby(["horizon_bars", "settlement_time", "side"], dropna=False)["signal_id"].transform("count")
    aggregate_same_direction = ordered.copy()
    aggregate_same_direction["source_signal_count"] = same_direction_counts
    aggregate_same_direction = aggregate_same_direction.groupby(["horizon_bars", "settlement_time", "side"], as_index=False, dropna=False).head(1).copy()
    aggregate_same_direction["concurrency_mode"] = "aggregate_same_direction"

    return pd.concat([independent, first_signal, aggregate_same_direction], ignore_index=True, sort=False)


def add_pricing_results(binary_bets: pd.DataFrame, buy_prices: Iterable[float], fee_drags: Iterable[float] | None = None, fixed_stake: float = FIXED_STAKE) -> pd.DataFrame:
    fee_values = list(fee_drags or FEE_DRAGS)
    rows: list[pd.DataFrame] = []
    for buy_price, fee_drag in itertools.product(buy_prices, fee_values):
        priced = binary_bets.copy()
        cost_per_share = float(buy_price) * (1.0 + float(fee_drag))
        shares = fixed_stake / cost_per_share
        gross_payout = np.where(priced["won"].astype(bool), shares * 1.0, 0.0)
        priced["buy_price"] = float(buy_price)
        priced["fee_drag"] = float(fee_drag)
        priced["stake"] = fixed_stake
        priced["shares"] = shares
        priced["cost"] = fixed_stake
        priced["gross_payout"] = gross_payout
        priced["pnl"] = gross_payout - fixed_stake
        priced["return_on_stake"] = priced["pnl"] / fixed_stake
        rows.append(priced)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_equity_curve(priced_bets: pd.DataFrame, initial_bankroll: float = INITIAL_BANKROLL) -> pd.DataFrame:
    if priced_bets.empty:
        return pd.DataFrame(columns=["settlement_time", "pnl", "equity", "drawdown"])
    ordered = priced_bets.sort_values(["settlement_time", "signal_id", "horizon_bars"]).copy()
    curve = ordered.groupby("settlement_time", as_index=False).agg(pnl=("pnl", "sum"), bet_count=("signal_id", "count"))
    curve["equity"] = initial_bankroll + curve["pnl"].cumsum()
    curve["roll_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["roll_max"] - 1.0
    return curve
