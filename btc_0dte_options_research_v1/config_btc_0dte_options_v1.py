from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(r"D:\workspace\20260325\btc_0dte_options_research_v1")
WORKSPACE_DIR = Path(r"D:\workspace\20260325")
DATA_DIR = WORKSPACE_DIR / "data"
POLY_RESEARCH_DIR = WORKSPACE_DIR / "polymarket_multi_horizon_research_v1"
POLY_DATA_DIR = POLY_RESEARCH_DIR / "data_poly"

BTC_15M_FILE = DATA_DIR / "BTCUSDT_15m.csv"
V17_SIGNALS_FILE = POLY_DATA_DIR / "signals_v17_v1.csv"

DATA_SYNTHETIC_DIR = BASE_DIR / "data_synthetic_options"
OUTPUT_DIR = BASE_DIR / "output_btc_0dte_options_v1"
SYNTHETIC_OPTION_TRADES_FILE = DATA_SYNTHETIC_DIR / "synthetic_btc_0dte_option_trades_worst_case_v1.csv"
SYNTHETIC_OPTION_QUOTES_FILE = DATA_SYNTHETIC_DIR / "synthetic_btc_0dte_option_quotes_worst_case_v1.csv"

START_DATE = "2025-07-01 00:00:00"
END_DATE = "2026-05-01 23:59:59"
SIGNAL_VERSION = "v17"
TIMEFRAME_MINUTES = 15
HOLD_HOURS = 6.0
ENTRY_DELAY_MINUTES = 15
EXPIRY_HOUR_UTC = 8
MIN_TTE_HOURS = HOLD_HOURS + 0.25
MAX_TTE_HOURS = 24.0
STRIKE_STEP_USD = 500.0

ROLLING_VOL_BARS_FAST = 96
ROLLING_VOL_BARS_SLOW = 672
ANNUALIZATION_BARS = 365 * 24 * 4
ENTRY_IV_MULTIPLIER = 1.45
ENTRY_IV_ADDON = 0.25
ENTRY_IV_FLOOR = 0.90
ENTRY_IV_CAP = 3.50
EXIT_IV_MULTIPLIER = 0.60
EXIT_IV_SUBTRACT = 0.20
EXIT_IV_FLOOR = 0.25
EXIT_IV_CAP = 2.50
RISK_FREE_RATE = 0.0

BASE_QUOTE_SPREAD_RATE = 0.10
NEAR_EXPIRY_SPREAD_RATE = 0.18
OTM_SPREAD_RATE = 0.12
ENTRY_EXTRA_MARKUP_RATE = 0.06
EXIT_EXTRA_HAIRCUT_RATE = 0.08
INTRINSIC_BID_FLOOR_RATE = 0.985
MIN_OPTION_PRICE_USD = 1.0

BINANCE_OPTION_TAKER_FEE_RATE = 0.0003
BINANCE_OPTION_FEE_CAP_RATE = 0.10
BINANCE_ENTRY_SLIPPAGE_RATE = 0.01
BINANCE_EXIT_SLIPPAGE_RATE = 0.01
FIXED_PREMIUM_STAKE_USD = 100.0
INITIAL_BANKROLL = 10_000.0
MAX_OPEN_POSITIONS = 1

VARIANTS = {
    "all_signals": {"sides": None, "entry_ids": None},
    "long_only": {"sides": ("long",), "entry_ids": None},
    "short_only": {"sides": ("short",), "entry_ids": None},
    "long_quality": {"sides": ("long",), "entry_ids": ("Long1", "Long2", "TrendLong")},
    "deep_entries": {"sides": None, "entry_ids": ("Long2", "Long3", "Short2", "Short3")},
}
