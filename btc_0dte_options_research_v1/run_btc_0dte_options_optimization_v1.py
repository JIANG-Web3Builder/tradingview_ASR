from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_btc_0dte_options_v1 import (
    BINANCE_ENTRY_SLIPPAGE_RATE,
    BINANCE_EXIT_SLIPPAGE_RATE,
    BINANCE_OPTION_FEE_CAP_RATE,
    BINANCE_OPTION_TAKER_FEE_RATE,
    ENTRY_DELAY_MINUTES,
    EXPIRY_HOUR_UTC,
    FIXED_PREMIUM_STAKE_USD,
    INITIAL_BANKROLL,
    MAX_OPEN_POSITIONS,
    OUTPUT_DIR,
    STRIKE_STEP_USD,
    VARIANTS,
)
from data_loader_btc_0dte_options_v1 import load_btc_15m, load_v17_signals, validate_btc_15m
from synthetic_option_pricing_v1 import add_realized_volatility, black_scholes_price, option_intrinsic_value

OPT_OUTPUT_DIR = OUTPUT_DIR / "optimization_v1"


@dataclass(frozen=True)
class PricingRegime:
    name: str
    entry_iv_mult: float
    entry_iv_add: float
    entry_iv_floor: float
    entry_iv_cap: float
    exit_iv_mult: float
    exit_iv_subtract: float
    exit_iv_floor: float
    exit_iv_cap: float
    base_spread: float
    near_expiry_spread: float
    otm_spread: float
    entry_markup: float
    exit_haircut: float
    entry_slippage: float
    exit_slippage: float


@dataclass(frozen=True)
class Policy:
    name: str
    variant: str
    otm_pct: float
    max_hold_hours: float
    take_profit: float | None
    stop_loss: float | None
    min_tte_hours: float
    max_tte_hours: float
    min_rv_quantile: float | None
    min_direction_edge_pct: float | None
    pricing_regime: str


PRICING_REGIMES = {
    "worst": PricingRegime("worst", 1.45, 0.25, 0.90, 3.50, 0.60, 0.20, 0.25, 2.50, 0.10, 0.18, 0.12, 0.06, 0.08, 0.01, 0.01),
    "harsh": PricingRegime("harsh", 1.25, 0.15, 0.75, 3.00, 0.75, 0.10, 0.30, 2.50, 0.07, 0.12, 0.08, 0.035, 0.05, 0.0075, 0.0075),
    "conservative": PricingRegime("conservative", 1.10, 0.08, 0.60, 2.50, 0.90, 0.05, 0.35, 2.50, 0.05, 0.08, 0.05, 0.02, 0.03, 0.005, 0.005),
}

POLICIES = [
    Policy("baseline_worst_atm_6h", "all_signals", 0.00, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("deep_worst_atm_6h", "deep_entries", 0.00, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_atm_6h", "long_quality", 0.00, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_itm1_6h", "long_quality", -0.01, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_itm2_6h", "long_quality", -0.02, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_otm1_6h", "long_quality", 0.01, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_otm2_6h", "long_quality", 0.02, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("deep_worst_itm1_6h", "deep_entries", -0.01, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("deep_worst_otm1_6h", "deep_entries", 0.01, 6.0, None, None, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_atm_3h", "long_quality", 0.00, 3.0, None, None, 3.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_atm_4h", "long_quality", 0.00, 4.0, None, None, 4.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_atm_tp50_sl50", "long_quality", 0.00, 6.0, 0.50, -0.50, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_atm_tp75_sl50", "long_quality", 0.00, 6.0, 0.75, -0.50, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_atm_tp100_sl60", "long_quality", 0.00, 6.0, 1.00, -0.60, 6.25, 24.0, None, None, "worst"),
    Policy("long_quality_worst_atm_tte8_20", "long_quality", 0.00, 6.0, None, None, 8.0, 20.0, None, None, "worst"),
    Policy("long_quality_worst_atm_rv70", "long_quality", 0.00, 6.0, None, None, 6.25, 24.0, 0.70, None, "worst"),
    Policy("long_quality_worst_atm_edge50", "long_quality", 0.00, 6.0, None, None, 6.25, 24.0, None, 0.50, "worst"),
    Policy("long_quality_worst_atm_rv70_edge50", "long_quality", 0.00, 6.0, None, None, 6.25, 24.0, 0.70, 0.50, "worst"),
    Policy("deep_worst_atm_tp50_sl50", "deep_entries", 0.00, 6.0, 0.50, -0.50, 6.25, 24.0, None, None, "worst"),
    Policy("deep_worst_atm_rv70", "deep_entries", 0.00, 6.0, None, None, 6.25, 24.0, 0.70, None, "worst"),
    Policy("long_quality_harsh_atm_6h", "long_quality", 0.00, 6.0, None, None, 6.25, 24.0, None, None, "harsh"),
    Policy("long_quality_harsh_itm1_tp50_sl50", "long_quality", -0.01, 6.0, 0.50, -0.50, 6.25, 24.0, None, None, "harsh"),
    Policy("long_quality_harsh_atm_rv70_edge50", "long_quality", 0.00, 6.0, None, None, 6.25, 24.0, 0.70, 0.50, "harsh"),
    Policy("deep_harsh_atm_6h", "deep_entries", 0.00, 6.0, None, None, 6.25, 24.0, None, None, "harsh"),
    Policy("deep_harsh_itm1_tp50_sl50", "deep_entries", -0.01, 6.0, 0.50, -0.50, 6.25, 24.0, None, None, "harsh"),
    Policy("long_quality_conservative_atm_6h", "long_quality", 0.00, 6.0, None, None, 6.25, 24.0, None, None, "conservative"),
    Policy("long_quality_conservative_itm1_tp50_sl50", "long_quality", -0.01, 6.0, 0.50, -0.50, 6.25, 24.0, None, None, "conservative"),
    Policy("long_quality_conservative_atm_rv70_edge50", "long_quality", 0.00, 6.0, None, None, 6.25, 24.0, 0.70, 0.50, "conservative"),
    Policy("deep_conservative_atm_6h", "deep_entries", 0.00, 6.0, None, None, 6.25, 24.0, None, None, "conservative"),
    Policy("deep_conservative_itm1_tp50_sl50", "deep_entries", -0.01, 6.0, 0.50, -0.50, 6.25, 24.0, None, None, "conservative"),
]


def build_policy_grid() -> list[Policy]:
    policies: dict[str, Policy] = {policy.name: policy for policy in POLICIES}
    variants = ["all_signals", "long_only", "short_only", "long_quality", "deep_entries"]
    regimes = ["worst", "harsh", "conservative"]
    strikes = [(-0.02, "itm2"), (-0.01, "itm1"), (0.00, "atm"), (0.01, "otm1"), (0.02, "otm2")]
    holds = [(3.0, "3h"), (4.0, "4h"), (6.0, "6h")]
    exits = [(None, None, "time"), (0.50, -0.50, "tp50_sl50"), (0.75, -0.50, "tp75_sl50"), (1.00, -0.60, "tp100_sl60")]
    filters = [(None, None, "nofilter"), (0.70, None, "rv70"), (None, 0.10, "edge10"), (None, 0.20, "edge20"), (0.70, 0.10, "rv70_edge10")]
    for variant in variants:
        for regime in regimes:
            for otm_pct, strike_name in strikes:
                for hold_hours, hold_name in holds:
                    for take_profit, stop_loss, exit_name in exits:
                        for rv_quantile, edge_pct, filter_name in filters:
                            min_tte = hold_hours + 0.25
                            name = f"{variant}_{regime}_{strike_name}_{hold_name}_{exit_name}_{filter_name}"
                            policies[name] = Policy(name, variant, otm_pct, hold_hours, take_profit, stop_loss, min_tte, 24.0, rv_quantile, edge_pct, regime)
    return list(policies.values())


def nearest_price_row(price_data: pd.DataFrame, target_time: pd.Timestamp) -> pd.Series | None:
    eligible = price_data[price_data["open_time"] >= target_time]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def next_expiry(entry_time: pd.Timestamp, min_tte_hours: float, max_tte_hours: float) -> pd.Timestamp | None:
    entry_time = pd.Timestamp(entry_time)
    expiry = entry_time.normalize() + pd.Timedelta(hours=EXPIRY_HOUR_UTC)
    if expiry <= entry_time + pd.Timedelta(hours=min_tte_hours):
        expiry += pd.Timedelta(days=1)
    tte_hours = (expiry - entry_time).total_seconds() / 3600.0
    if tte_hours < min_tte_hours or tte_hours > max_tte_hours:
        return None
    return expiry


def choose_option_type(side: str) -> str:
    return "call" if str(side).lower() == "long" else "put"


def directional_strike(spot: float, option_type: str, otm_pct: float) -> float:
    if option_type == "call":
        raw = float(spot) * (1.0 + float(otm_pct))
    else:
        raw = float(spot) * (1.0 - float(otm_pct))
    return max(STRIKE_STEP_USD, round(raw / STRIKE_STEP_USD) * STRIKE_STEP_USD)


def iv_for_phase(rv_base: float, regime: PricingRegime, phase: str) -> float:
    if phase == "entry":
        return min(max(float(rv_base) * regime.entry_iv_mult + regime.entry_iv_add, regime.entry_iv_floor), regime.entry_iv_cap)
    return min(max(float(rv_base) * regime.exit_iv_mult - regime.exit_iv_subtract, regime.exit_iv_floor), regime.exit_iv_cap)


def spread_rate(spot: float, strike: float, tte_hours: float, regime: PricingRegime) -> float:
    moneyness_gap = abs(float(spot) / float(strike) - 1.0) if strike else 0.0
    spread = regime.base_spread
    if tte_hours <= 8.0:
        spread += regime.near_expiry_spread
    if moneyness_gap >= 0.015:
        spread += regime.otm_spread
    return max(spread, 0.0)


def option_quote(spot: float, strike: float, expiry_time: pd.Timestamp, quote_time: pd.Timestamp, rv_base: float, option_type: str, phase: str, regime: PricingRegime) -> dict:
    tte_hours = max((pd.Timestamp(expiry_time) - pd.Timestamp(quote_time)).total_seconds() / 3600.0, 0.0)
    t_years = max(tte_hours / (365.0 * 24.0), 1.0 / (365.0 * 24.0 * 60.0))
    iv = iv_for_phase(rv_base, regime, phase)
    theoretical = black_scholes_price(spot, strike, t_years, iv, option_type)
    intrinsic = option_intrinsic_value(spot, strike, option_type)
    spread = spread_rate(spot, strike, tte_hours, regime)
    mid = max(theoretical, intrinsic, 1.0)
    ask = max(mid * (1.0 + spread / 2.0), intrinsic + 1.0)
    bid = max(min(mid * (1.0 - spread / 2.0), ask), intrinsic * 0.985)
    if phase == "entry":
        executable = ask * (1.0 + regime.entry_markup)
    else:
        executable = max(bid * (1.0 - regime.exit_haircut), intrinsic * 0.985, 0.0)
    return {
        "quote_time": pd.Timestamp(quote_time),
        "spot": float(spot),
        "tte_hours": float(tte_hours),
        "iv_used": float(iv),
        "theoretical_mid": float(theoretical),
        "intrinsic": float(intrinsic),
        "bid": float(bid),
        "ask": float(ask),
        "worst_executable_price": float(executable),
    }


def execution_values(entry_price: float, exit_price: float, entry_spot: float, exit_spot: float, regime: PricingRegime) -> dict:
    entry_after_slippage = entry_price * (1.0 + regime.entry_slippage)
    exit_after_slippage = max(exit_price * (1.0 - regime.exit_slippage), 0.0)
    contracts = FIXED_PREMIUM_STAKE_USD / entry_after_slippage if entry_after_slippage > 0 else 0.0
    entry_premium_value = contracts * entry_after_slippage
    exit_premium_value = contracts * exit_after_slippage
    entry_notional = contracts * float(entry_spot)
    exit_notional = contracts * float(exit_spot)
    entry_fee = min(entry_notional * BINANCE_OPTION_TAKER_FEE_RATE, entry_premium_value * BINANCE_OPTION_FEE_CAP_RATE)
    exit_fee = min(exit_notional * BINANCE_OPTION_TAKER_FEE_RATE, exit_premium_value * BINANCE_OPTION_FEE_CAP_RATE)
    total_cost = entry_premium_value + entry_fee
    pnl = exit_premium_value - exit_fee - total_cost
    return_on_cost = pnl / total_cost if total_cost > 0 else np.nan
    return {
        "entry_price_after_slippage": float(entry_after_slippage),
        "exit_price_after_slippage": float(exit_after_slippage),
        "contracts": float(contracts),
        "entry_underlying_notional": float(entry_notional),
        "exit_underlying_notional": float(exit_notional),
        "entry_premium_value": float(entry_premium_value),
        "exit_premium_value": float(exit_premium_value),
        "entry_fee": float(entry_fee),
        "exit_fee": float(exit_fee),
        "total_cost": float(total_cost),
        "pnl": float(pnl),
        "return_on_cost": float(return_on_cost),
    }


def add_signal_filters(signals: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
    signal_times = signals["entry_time"].copy()
    price_lookup = price_data.set_index("open_time")
    future_24 = price_data[["open_time", "close"]].copy()
    future_24["future_6h_close"] = future_24["close"].shift(-24)
    future_24 = future_24.set_index("open_time")
    enriched = signals.copy()
    rv_series = price_lookup["rv_base"]
    close_series = price_lookup["close"]
    enriched["entry_rv_base_for_filter"] = [float(rv_series.asof(t)) if pd.notna(rv_series.asof(t)) else np.nan for t in signal_times]
    entry_close = [float(close_series.asof(t)) if pd.notna(close_series.asof(t)) else np.nan for t in signal_times]
    future_close = [float(future_24["future_6h_close"].asof(t)) if pd.notna(future_24["future_6h_close"].asof(t)) else np.nan for t in signal_times]
    sign = np.where(enriched["side"].astype(str) == "long", 1.0, -1.0)
    enriched["realized_6h_direction_return_pct"] = (np.array(future_close) / np.array(entry_close) - 1.0) * 100.0 * sign
    enriched["historical_entry_edge_pct"] = np.nan
    grouped = enriched.sort_values(["entry_time", "signal_id"]).groupby(["side", "entry_id"], dropna=False)
    for _, idx in grouped.groups.items():
        values = enriched.loc[idx, "realized_6h_direction_return_pct"].shift(1).expanding(min_periods=20).mean()
        enriched.loc[idx, "historical_entry_edge_pct"] = values
    return enriched


def apply_variant(signals: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    variant = VARIANTS[variant_name]
    filtered = signals.copy()
    sides = variant.get("sides")
    entry_ids = variant.get("entry_ids")
    if sides is not None:
        filtered = filtered[filtered["side"].isin(sides)].copy()
    if entry_ids is not None:
        filtered = filtered[filtered["entry_id"].isin(entry_ids)].copy()
    return filtered


def apply_policy_filters(signals: pd.DataFrame, policy: Policy, rv_thresholds: dict[float, float]) -> pd.DataFrame:
    filtered = apply_variant(signals, policy.variant)
    if policy.min_rv_quantile is not None:
        threshold = rv_thresholds[policy.min_rv_quantile]
        filtered = filtered[filtered["entry_rv_base_for_filter"] >= threshold].copy()
    if policy.min_direction_edge_pct is not None:
        filtered = filtered[filtered["historical_entry_edge_pct"] >= policy.min_direction_edge_pct].copy()
    return filtered.sort_values(["entry_time", "signal_id"]).reset_index(drop=True)


def build_trade_for_signal(signal: pd.Series, price_data: pd.DataFrame, policy: Policy, regime: PricingRegime) -> dict | None:
    signal_time = pd.Timestamp(signal["entry_time"])
    entry_row = nearest_price_row(price_data, signal_time + pd.Timedelta(minutes=ENTRY_DELAY_MINUTES))
    if entry_row is None:
        return None
    entry_time = pd.Timestamp(entry_row["open_time"])
    expiry_time = next_expiry(entry_time, policy.min_tte_hours, policy.max_tte_hours)
    if expiry_time is None:
        return None
    max_exit_time = min(entry_time + pd.Timedelta(hours=policy.max_hold_hours), expiry_time - pd.Timedelta(minutes=15))
    if max_exit_time <= entry_time:
        return None

    option_type = choose_option_type(str(signal["side"]))
    strike = directional_strike(float(entry_row["close"]), option_type, policy.otm_pct)
    entry_quote = option_quote(float(entry_row["close"]), strike, expiry_time, entry_time, float(entry_row["rv_base"]), option_type, "entry", regime)
    entry_execution = execution_values(entry_quote["worst_executable_price"], entry_quote["worst_executable_price"], float(entry_row["close"]), float(entry_row["close"]), regime)
    entry_total_cost = entry_execution["total_cost"]

    path = price_data[(price_data["open_time"] > entry_time) & (price_data["open_time"] <= max_exit_time)].copy()
    if path.empty:
        return None

    selected_exit = None
    selected_exit_quote = None
    selected_execution = None
    exit_reason = "time"
    for path_row in path.itertuples(index=False):
        quote_time = pd.Timestamp(path_row.open_time)
        exit_quote = option_quote(float(path_row.close), strike, expiry_time, quote_time, float(path_row.rv_base), option_type, "exit", regime)
        execution = execution_values(entry_quote["worst_executable_price"], exit_quote["worst_executable_price"], float(entry_row["close"]), float(path_row.close), regime)
        roc = execution["pnl"] / entry_total_cost if entry_total_cost > 0 else np.nan
        selected_exit = path_row
        selected_exit_quote = exit_quote
        selected_execution = execution
        if policy.take_profit is not None and roc >= policy.take_profit:
            exit_reason = "take_profit"
            break
        if policy.stop_loss is not None and roc <= policy.stop_loss:
            exit_reason = "stop_loss"
            break

    if selected_exit is None or selected_exit_quote is None or selected_execution is None:
        return None

    exit_spot = float(selected_exit.close)
    direction_return_pct = (exit_spot / float(entry_row["close"]) - 1.0) * 100.0 * (1.0 if str(signal["side"]) == "long" else -1.0)
    return {
        "policy": policy.name,
        "variant": policy.variant,
        "pricing_regime": policy.pricing_regime,
        "signal_id": int(signal["signal_id"]),
        "trade_id": int(signal["trade_id"]),
        "side": str(signal["side"]),
        "entry_id": str(signal["entry_id"]),
        "signal_time": signal_time,
        "entry_time": entry_time,
        "exit_time": pd.Timestamp(selected_exit.open_time),
        "expiry_time": expiry_time,
        "exit_reason": exit_reason,
        "option_type": option_type,
        "otm_pct": float(policy.otm_pct),
        "strike": float(strike),
        "entry_spot": float(entry_row["close"]),
        "exit_spot": exit_spot,
        "direction_return_pct": float(direction_return_pct),
        "entry_tte_hours": float(entry_quote["tte_hours"]),
        "exit_tte_hours": float(selected_exit_quote["tte_hours"]),
        "entry_iv_used": float(entry_quote["iv_used"]),
        "exit_iv_used": float(selected_exit_quote["iv_used"]),
        "entry_theoretical_mid": float(entry_quote["theoretical_mid"]),
        "exit_theoretical_mid": float(selected_exit_quote["theoretical_mid"]),
        "entry_intrinsic": float(entry_quote["intrinsic"]),
        "exit_intrinsic": float(selected_exit_quote["intrinsic"]),
        "entry_worst_model_price": float(entry_quote["worst_executable_price"]),
        "exit_worst_model_price": float(selected_exit_quote["worst_executable_price"]),
        "historical_entry_edge_pct": float(signal["historical_entry_edge_pct"]) if pd.notna(signal["historical_entry_edge_pct"]) else np.nan,
        "entry_rv_base_for_filter": float(signal["entry_rv_base_for_filter"]) if pd.notna(signal["entry_rv_base_for_filter"]) else np.nan,
        **selected_execution,
        "won": bool(selected_execution["pnl"] > 0),
    }


def execute_with_concurrency(candidates: list[dict]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    ordered = sorted(candidates, key=lambda x: (x["entry_time"], x["signal_id"]))
    rows: list[dict] = []
    open_positions: list[tuple[pd.Timestamp, float]] = []
    bankroll = INITIAL_BANKROLL
    for row in ordered:
        entry_time = pd.Timestamp(row["entry_time"])
        still_open: list[tuple[pd.Timestamp, float]] = []
        for exit_time, pnl in open_positions:
            if exit_time <= entry_time:
                bankroll += pnl
            else:
                still_open.append((exit_time, pnl))
        open_positions = still_open
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue
        if bankroll < float(row["total_cost"]):
            continue
        rows.append(row)
        open_positions.append((pd.Timestamp(row["exit_time"]), float(row["pnl"])))
    return pd.DataFrame(rows)


def equity_curve(trades: pd.DataFrame, policy_name: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["policy", "exit_time", "pnl", "equity", "drawdown", "trade_count"])
    curve = trades.sort_values(["exit_time", "signal_id"]).groupby("exit_time", as_index=False).agg(pnl=("pnl", "sum"), trade_count=("signal_id", "count"))
    curve["policy"] = policy_name
    curve["equity"] = INITIAL_BANKROLL + curve["pnl"].cumsum()
    curve["roll_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["roll_max"] - 1.0
    return curve[["policy", "exit_time", "pnl", "equity", "drawdown", "trade_count"]]


def profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return np.nan
    gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def longest_loss_streak(trades: pd.DataFrame) -> int:
    longest = 0
    current = 0
    for won in trades.sort_values(["exit_time", "signal_id"])["won"].astype(bool):
        if won:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def summarize_policy(policy: Policy, candidate_count: int, trades: pd.DataFrame, curve: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "policy": policy.name,
            "variant": policy.variant,
            "pricing_regime": policy.pricing_regime,
            "otm_pct": policy.otm_pct,
            "max_hold_hours": policy.max_hold_hours,
            "take_profit": policy.take_profit,
            "stop_loss": policy.stop_loss,
            "min_tte_hours": policy.min_tte_hours,
            "max_tte_hours": policy.max_tte_hours,
            "min_rv_quantile": policy.min_rv_quantile,
            "min_direction_edge_pct": policy.min_direction_edge_pct,
            "candidate_count": int(candidate_count),
            "trade_count": 0,
            "trade_rate": 0.0,
            "win_rate": np.nan,
            "total_cost": 0.0,
            "total_pnl": 0.0,
            "roi_on_cost": np.nan,
            "final_equity": INITIAL_BANKROLL,
            "max_drawdown": np.nan,
            "profit_factor": np.nan,
            "avg_return_on_cost": np.nan,
            "median_return_on_cost": np.nan,
            "longest_loss_streak": 0,
            "tp_count": 0,
            "sl_count": 0,
            "time_exit_count": 0,
        }
    total_cost = float(trades["total_cost"].sum())
    total_pnl = float(trades["pnl"].sum())
    return {
        "policy": policy.name,
        "variant": policy.variant,
        "pricing_regime": policy.pricing_regime,
        "otm_pct": policy.otm_pct,
        "max_hold_hours": policy.max_hold_hours,
        "take_profit": policy.take_profit,
        "stop_loss": policy.stop_loss,
        "min_tte_hours": policy.min_tte_hours,
        "max_tte_hours": policy.max_tte_hours,
        "min_rv_quantile": policy.min_rv_quantile,
        "min_direction_edge_pct": policy.min_direction_edge_pct,
        "candidate_count": int(candidate_count),
        "trade_count": int(len(trades)),
        "trade_rate": float(len(trades) / candidate_count) if candidate_count else 0.0,
        "win_rate": float(trades["won"].mean()),
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "roi_on_cost": float(total_pnl / total_cost) if total_cost else np.nan,
        "final_equity": float(INITIAL_BANKROLL + total_pnl),
        "max_drawdown": float(curve["drawdown"].min()) if not curve.empty else np.nan,
        "profit_factor": profit_factor(trades),
        "avg_return_on_cost": float(trades["return_on_cost"].mean()),
        "median_return_on_cost": float(trades["return_on_cost"].median()),
        "longest_loss_streak": longest_loss_streak(trades),
        "tp_count": int((trades["exit_reason"] == "take_profit").sum()),
        "sl_count": int((trades["exit_reason"] == "stop_loss").sum()),
        "time_exit_count": int((trades["exit_reason"] == "time").sum()),
    }


def run_policy(policy: Policy, signals: pd.DataFrame, price_data: pd.DataFrame, rv_thresholds: dict[float, float]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    regime = PRICING_REGIMES[policy.pricing_regime]
    policy_signals = apply_policy_filters(signals, policy, rv_thresholds)
    candidate_rows: list[dict] = []
    for _, signal in policy_signals.iterrows():
        row = build_trade_for_signal(signal, price_data, policy, regime)
        if row is not None:
            candidate_rows.append(row)
    trades = execute_with_concurrency(candidate_rows)
    curve = equity_curve(trades, policy.name)
    summary = summarize_policy(policy, len(candidate_rows), trades, curve)
    return summary, trades, curve


def plot_top_curves(summary: pd.DataFrame, curves: pd.DataFrame, output_path: Path) -> None:
    if summary.empty or curves.empty:
        return
    selected = summary.sort_values(["total_pnl", "max_drawdown"], ascending=[False, False]).head(8)["policy"].tolist()
    fig, ax = plt.subplots(figsize=(13, 7))
    for policy, group in curves[curves["policy"].isin(selected)].groupby("policy"):
        ax.plot(pd.to_datetime(group["exit_time"]), group["equity"], label=policy, linewidth=1.6)
    ax.axhline(INITIAL_BANKROLL, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_title("BTC 0DTE Option Optimization Top Equity Curves")
    ax.set_xlabel("Exit Time")
    ax.set_ylabel("Equity USD")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(summary: pd.DataFrame, validation: dict, output_path: Path) -> None:
    ordered = summary.sort_values("total_pnl", ascending=False)
    active_ordered = ordered[ordered["trade_count"] > 0].copy()
    lines = [
        "# BTC 0DTE Options Optimization v1",
        "",
        "## 数据校验",
        "",
        f"- BTC rows: {validation['rows']}",
        f"- BTC range: {validation['start']} to {validation['end']}",
        f"- Bad 15m spacing count: {validation['bad_15m_spacing_count']}",
        f"- Policies scanned: {len(summary)}",
        f"- Active policies with trades: {len(active_ordered)}",
        "",
        "## Top Policies by PnL",
        "",
        "| Policy | Regime | Variant | Trades | Win | ROI | PnL | Final equity | Max DD | PF | TP | SL | Time |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in active_ordered.head(15).itertuples(index=False):
        win = "nan" if pd.isna(row.win_rate) else f"{row.win_rate:.2%}"
        roi = "nan" if pd.isna(row.roi_on_cost) else f"{row.roi_on_cost:.2%}"
        dd = "nan" if pd.isna(row.max_drawdown) else f"{row.max_drawdown:.2%}"
        pf = "nan" if pd.isna(row.profit_factor) else f"{row.profit_factor:.3f}"
        lines.append(f"| {row.policy} | {row.pricing_regime} | {row.variant} | {row.trade_count} | {win} | {roi} | {row.total_pnl:.2f} | {row.final_equity:.2f} | {dd} | {pf} | {row.tp_count} | {row.sl_count} | {row.time_exit_count} |")
    lines.extend([
        "",
        "## Bottom Policies by PnL",
        "",
        "| Policy | Regime | Variant | Trades | Win | ROI | PnL | Max DD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in ordered.tail(10).itertuples(index=False):
        win = "nan" if pd.isna(row.win_rate) else f"{row.win_rate:.2%}"
        roi = "nan" if pd.isna(row.roi_on_cost) else f"{row.roi_on_cost:.2%}"
        dd = "nan" if pd.isna(row.max_drawdown) else f"{row.max_drawdown:.2%}"
        lines.append(f"| {row.policy} | {row.pricing_regime} | {row.variant} | {row.trade_count} | {win} | {roi} | {row.total_pnl:.2f} | {dd} |")
    lines.extend([
        "",
        "## 结论提示",
        "",
        "- 如果 worst regime 下仍全部亏损，说明在最恶劣成交假设下买方期权策略没有通过压力测试。",
        "- harsh/conservative regime 用来观察结果是否只被极端价格假设压垮，不代表真实盘口。",
        "- 真正可实盘的组合应同时满足正 PnL、足够交易数、低回撤，并且在 harsh 或 worst 下不崩溃。",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    btc_raw = load_btc_15m()
    validation = validate_btc_15m(btc_raw)
    btc = add_realized_volatility(btc_raw)
    signals = add_signal_filters(load_v17_signals(), btc)
    rv_thresholds = {0.70: float(signals["entry_rv_base_for_filter"].quantile(0.70))}
    policy_list = build_policy_grid()

    summaries: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    curve_frames: list[pd.DataFrame] = []
    for idx, policy in enumerate(policy_list, start=1):
        summary, trades, curve = run_policy(policy, signals, btc, rv_thresholds)
        summaries.append(summary)
        if not trades.empty:
            trade_frames.append(trades)
        if not curve.empty:
            curve_frames.append(curve)
        if idx == 1 or idx % 100 == 0 or idx == len(policy_list):
            print(f"Scanned {idx}/{len(policy_list)} policies; latest={policy.name}; trades={summary['trade_count']} pnl={summary['total_pnl']:.2f}")

    summary_df = pd.DataFrame(summaries).sort_values("total_pnl", ascending=False).reset_index(drop=True)
    baseline = summary_df.loc[summary_df["policy"] == "baseline_worst_atm_6h"]
    if not baseline.empty:
        baseline_pnl = float(baseline.iloc[0]["total_pnl"])
        baseline_roi = float(baseline.iloc[0]["roi_on_cost"])
        summary_df["pnl_vs_baseline"] = summary_df["total_pnl"] - baseline_pnl
        summary_df["roi_vs_baseline"] = summary_df["roi_on_cost"] - baseline_roi
    trades_df = pd.concat(trade_frames, ignore_index=True, sort=False) if trade_frames else pd.DataFrame()
    curves_df = pd.concat(curve_frames, ignore_index=True, sort=False) if curve_frames else pd.DataFrame()
    active_summary_df = summary_df[summary_df["trade_count"] > 0].copy()

    summary_df.to_csv(OPT_OUTPUT_DIR / "btc_0dte_option_optimization_summary_v1.csv", index=False)
    trades_df.to_csv(OPT_OUTPUT_DIR / "btc_0dte_option_optimization_trades_v1.csv", index=False)
    curves_df.to_csv(OPT_OUTPUT_DIR / "btc_0dte_option_optimization_equity_curves_v1.csv", index=False)
    (OPT_OUTPUT_DIR / "btc_0dte_option_optimization_summary_v1.json").write_text(
        json.dumps(
            {
                "data_validation": validation,
                "policy_count": int(len(summary_df)),
                "active_policy_count": int(len(active_summary_df)),
                "best_active_policy": active_summary_df.iloc[0].replace({np.nan: None, np.inf: "inf", -np.inf: "-inf"}).to_dict() if not active_summary_df.empty else {},
                "summary": summary_df.replace({np.nan: None, np.inf: "inf", -np.inf: "-inf"}).to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_top_curves(summary_df, curves_df, OPT_OUTPUT_DIR / "btc_0dte_option_optimization_top_equity_v1.png")
    write_report(summary_df, validation, OPT_OUTPUT_DIR / "btc_0dte_option_optimization_report_v1.md")
    print("\nTop 10 active policies:")
    print(active_summary_df.head(10).to_string(index=False))
    print(f"\nSaved optimization output: {OPT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
