from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config_btc_0dte_options_v1 import (
    ANNUALIZATION_BARS,
    BASE_QUOTE_SPREAD_RATE,
    ENTRY_EXTRA_MARKUP_RATE,
    ENTRY_IV_ADDON,
    ENTRY_IV_CAP,
    ENTRY_IV_FLOOR,
    ENTRY_IV_MULTIPLIER,
    EXIT_EXTRA_HAIRCUT_RATE,
    EXIT_IV_CAP,
    EXIT_IV_FLOOR,
    EXIT_IV_MULTIPLIER,
    EXIT_IV_SUBTRACT,
    EXPIRY_HOUR_UTC,
    INTRINSIC_BID_FLOOR_RATE,
    MAX_TTE_HOURS,
    MIN_OPTION_PRICE_USD,
    MIN_TTE_HOURS,
    NEAR_EXPIRY_SPREAD_RATE,
    OTM_SPREAD_RATE,
    RISK_FREE_RATE,
    ROLLING_VOL_BARS_FAST,
    ROLLING_VOL_BARS_SLOW,
    STRIKE_STEP_USD,
)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def black_scholes_price(spot: float, strike: float, t_years: float, sigma: float, option_type: str, rate: float = RISK_FREE_RATE) -> float:
    spot = max(float(spot), 1e-9)
    strike = max(float(strike), 1e-9)
    t_years = max(float(t_years), 1.0 / (365.0 * 24.0 * 60.0))
    sigma = max(float(sigma), 1e-6)
    option_type = str(option_type).lower()
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t_years) / (sigma * math.sqrt(t_years))
    d2 = d1 - sigma * math.sqrt(t_years)
    if option_type == "call":
        return max(0.0, spot * normal_cdf(d1) - strike * math.exp(-rate * t_years) * normal_cdf(d2))
    return max(0.0, strike * math.exp(-rate * t_years) * normal_cdf(-d2) - spot * normal_cdf(-d1))


def option_intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    if str(option_type).lower() == "call":
        return max(float(spot) - float(strike), 0.0)
    return max(float(strike) - float(spot), 0.0)


def add_realized_volatility(btc: pd.DataFrame) -> pd.DataFrame:
    df = btc.copy()
    df["log_return"] = np.log(df["close"].astype(float) / df["close"].astype(float).shift(1))
    fast_vol = df["log_return"].rolling(ROLLING_VOL_BARS_FAST, min_periods=max(12, ROLLING_VOL_BARS_FAST // 4)).std() * math.sqrt(ANNUALIZATION_BARS)
    slow_vol = df["log_return"].rolling(ROLLING_VOL_BARS_SLOW, min_periods=max(48, ROLLING_VOL_BARS_SLOW // 4)).std() * math.sqrt(ANNUALIZATION_BARS)
    ewma_vol = df["log_return"].ewm(span=ROLLING_VOL_BARS_FAST, adjust=False, min_periods=12).std() * math.sqrt(ANNUALIZATION_BARS)
    df["rv_fast"] = fast_vol
    df["rv_slow"] = slow_vol
    df["rv_ewma"] = ewma_vol
    df["rv_base"] = pd.concat([df["rv_fast"], df["rv_slow"], df["rv_ewma"]], axis=1).max(axis=1)
    df["rv_base"] = df["rv_base"].fillna(method="bfill").fillna(method="ffill").clip(lower=0.25, upper=3.0)
    return df


def next_daily_expiry(entry_time: pd.Timestamp) -> pd.Timestamp | None:
    entry_time = pd.Timestamp(entry_time)
    expiry = entry_time.normalize() + pd.Timedelta(hours=EXPIRY_HOUR_UTC)
    if expiry <= entry_time + pd.Timedelta(hours=MIN_TTE_HOURS):
        expiry = expiry + pd.Timedelta(days=1)
    tte_hours = (expiry - entry_time).total_seconds() / 3600.0
    if tte_hours < MIN_TTE_HOURS or tte_hours > MAX_TTE_HOURS:
        return None
    return expiry


def atm_strike(spot: float) -> float:
    return round(float(spot) / STRIKE_STEP_USD) * STRIKE_STEP_USD


def choose_option_type(side: str) -> str:
    return "call" if str(side).lower() == "long" else "put"


def conservative_iv(rv_base: float, phase: str) -> float:
    rv_base = float(rv_base)
    if phase == "entry":
        return min(max(rv_base * ENTRY_IV_MULTIPLIER + ENTRY_IV_ADDON, ENTRY_IV_FLOOR), ENTRY_IV_CAP)
    return min(max(rv_base * EXIT_IV_MULTIPLIER - EXIT_IV_SUBTRACT, EXIT_IV_FLOOR), EXIT_IV_CAP)


def quote_spread_rate(spot: float, strike: float, tte_hours: float) -> float:
    moneyness_gap = abs(float(spot) / float(strike) - 1.0) if strike else 0.0
    spread = BASE_QUOTE_SPREAD_RATE
    if tte_hours <= 8.0:
        spread += NEAR_EXPIRY_SPREAD_RATE
    if moneyness_gap >= 0.015:
        spread += OTM_SPREAD_RATE
    return max(spread, 0.0)


def conservative_option_quote(spot: float, strike: float, expiry_time: pd.Timestamp, quote_time: pd.Timestamp, rv_base: float, option_type: str, phase: str) -> dict:
    quote_time = pd.Timestamp(quote_time)
    expiry_time = pd.Timestamp(expiry_time)
    tte_hours = max((expiry_time - quote_time).total_seconds() / 3600.0, 0.0)
    t_years = max(tte_hours / (365.0 * 24.0), 1.0 / (365.0 * 24.0 * 60.0))
    iv = conservative_iv(rv_base, phase)
    theoretical = black_scholes_price(spot, strike, t_years, iv, option_type)
    intrinsic = option_intrinsic_value(spot, strike, option_type)
    spread = quote_spread_rate(spot, strike, tte_hours)
    mid = max(theoretical, intrinsic, MIN_OPTION_PRICE_USD)
    ask = max(mid * (1.0 + spread / 2.0), intrinsic + MIN_OPTION_PRICE_USD)
    bid = max(mid * (1.0 - spread / 2.0), intrinsic * INTRINSIC_BID_FLOOR_RATE)
    bid = min(bid, ask)
    if phase == "entry":
        executable = ask * (1.0 + ENTRY_EXTRA_MARKUP_RATE)
    else:
        executable = max(bid * (1.0 - EXIT_EXTRA_HAIRCUT_RATE), intrinsic * INTRINSIC_BID_FLOOR_RATE, 0.0)
    return {
        "quote_time": quote_time,
        "expiry_time": expiry_time,
        "option_type": option_type,
        "strike": float(strike),
        "spot": float(spot),
        "tte_hours": float(tte_hours),
        "rv_base": float(rv_base),
        "iv_used": float(iv),
        "theoretical_mid": float(theoretical),
        "intrinsic": float(intrinsic),
        "spread_rate": float(spread),
        "bid": float(bid),
        "ask": float(ask),
        "worst_executable_price": float(max(executable, 0.0)),
        "quote_phase": phase,
    }
