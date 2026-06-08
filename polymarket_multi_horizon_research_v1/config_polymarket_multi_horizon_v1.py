from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(r"D:\workspace\20260325\polymarket_multi_horizon_research_v1")
WORKSPACE_DIR = Path(r"D:\workspace\20260325")
CHANNEL_RESEARCH_DIR = WORKSPACE_DIR / "asr_btc_channel_research_v1"
DATA_DIR = WORKSPACE_DIR / "data"
DATA_FILE_15M = DATA_DIR / "BTCUSDT_15m.csv"
DATA_POLY_DIR = BASE_DIR / "data_poly"
CACHED_DATA_FILE_15M = DATA_POLY_DIR / "BTCUSDT_15m_filtered_v1.csv"
CACHED_DATA_META_FILE_15M = DATA_POLY_DIR / "BTCUSDT_15m_filtered_meta_v1.json"
CACHED_INDICATOR_FILE = DATA_POLY_DIR / "BTCUSDT_15m_v17_indicators_v1.csv"
CACHED_SIGNALS_FILE = DATA_POLY_DIR / "signals_v17_v1.csv"
CACHED_SOURCE_EQUITY_FILE = DATA_POLY_DIR / "source_equity_v17_v1.csv"
CACHED_SIGNAL_META_FILE = DATA_POLY_DIR / "signals_v17_meta_v1.json"
OUTPUT_DIR = BASE_DIR / "output_polymarket_multi_horizon_v1"

START_DATE = "2024-01-01 00:00:00"
END_DATE = "2026-05-01 23:59:59"
SYMBOL = "BTCUSDT"
TIMEFRAME_MINUTES = 15
SIGNAL_VERSION = "v17"

HORIZON_BARS = [1, 2, 4, 8, 12, 24, 48, 96]
PRIMARY_HORIZON_BARS = [4, 24, 96]
BUY_PRICES = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
FEE_DRAGS = [0.00, 0.01, 0.02, 0.03, 0.05]
INITIAL_BANKROLL = 10_000.0
FIXED_STAKE = 100.0
KELLY_FRACTION_CAP = 0.05
TRAIN_END = "2025-07-01 00:00:00"
TIE_MODE = "loss"

ENTRY_IDS = [
    "Long1",
    "Long2",
    "Long3",
    "Short1",
    "Short2",
    "Short3",
    "TrendLong",
    "TrendShort",
    "RevLong",
    "RevShort",
]


@dataclass(frozen=True)
class StrategyVariant:
    name: str
    sides: tuple[str, ...] | None = None
    entry_ids: tuple[str, ...] | None = None
    max_horizon_bars: int | None = None
    min_horizon_bars: int | None = None


STRATEGY_VARIANTS = [
    StrategyVariant("all_signals"),
    StrategyVariant("long_only", sides=("long",)),
    StrategyVariant("short_only", sides=("short",)),
    StrategyVariant("long_60m_plus", sides=("long",), min_horizon_bars=4),
    StrategyVariant("short_60m_or_less", sides=("short",), max_horizon_bars=4),
    StrategyVariant("l1_only", entry_ids=("Long1", "Short1")),
    StrategyVariant("deep_entries", entry_ids=("Long2", "Long3", "Short2", "Short3")),
    StrategyVariant("trend_reversal", entry_ids=("TrendLong", "TrendShort", "RevLong", "RevShort")),
]
