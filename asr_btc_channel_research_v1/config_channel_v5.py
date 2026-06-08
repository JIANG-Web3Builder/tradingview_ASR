from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(r"D:\workspace\20260325\asr_btc_channel_research_v1")
OUTPUT_DIR = BASE_DIR / "output_channel_v5"
DATA_DIR = Path(r"D:\workspace\20260325\data")
DATA_FILE_15M = DATA_DIR / "BTCUSDT_15m.csv"
START_DATE = "2024-01-01 00:00:00"
END_DATE = "2026-05-01 23:59:59"
INITIAL_CAPITAL = 10000.0
COMMISSION_PCT = 0.036 / 100.0
SLIPPAGE_TICKS = 3.0
MINTICK = 0.01
SYMBOL = "BTCUSDT"
TIMEFRAME_MINUTES = 15

@dataclass(frozen=True)
class StrategyFeatures:
    version: str
    tp_mode: str
    tp_offset_multiplier: float | None
    long1_tp_uses_zone_offset: bool
    short1_tp_uses_zone_offset: bool
    reverse_breakeven_on_channel: bool
    reverse_lock_bars: int
    be_reentry_lock_bars: int | None
    deep_entry_overshoot_pct: float | None
    deep_entry_reclaim: bool
    mfe_zone_breakeven: bool
    level0_mfe_zone_breakeven: bool
    l1_reclaim_entry: bool
    l1_knife_half_size: bool
    no_progress_time_stop: bool
    channel_mode: str
    early_no_progress_time_stop: bool = False
    l1_adverse_fail_exit: bool = False
    short1_downtrend_filter: bool = False
    deep2_adverse_fail_exit: bool = False
    breakout_early_fail_exit: bool = False
    reverse_mfe_protect_during_lock: bool = False
    deep2_reclaim_fail_exit: bool = False
    breakout_no_progress_exit: bool = False

BASE_PARAMS = {
    "base_tf_minutes": 60,
    "hl_length_input": 75,
    "atr_length_input": 20,
    "vov_length_input": 100,
    "smooth_factor_input": 12,
    "base_width_pct": 8.0,
    "adjust_factor": 0.5,
    "zone_channel_pct": 8.0 / 100.0,
    "stop_mult": 2.5,
    "be_buffer": 0.3 / 100.0,
    "time_stop_bars_in": 24,
    "outside_stop_bars": 3,
    "trend_ma_input": 200,
    "enable_ma_entry_filter": False,
    "rsi_period": 14,
    "rsi_long": 40.0,
    "rsi_short": 60.0,
    "cooldown_bars_in": 3,
    "enable_short": True,
    "breakout_15m": True,
    "breakout_rsi_high": 78.0,
    "breakout_rsi_low": 22.0,
    "breakout_back_bars": 2,
    "breakout_confirm_bars": 2,
    "breakout_min_hold_bars": 8,
    "breakout_cooldown_bars": 12,
    "reverse_stop_lookback": 30,
    "alloc_pct1": 20.0,
    "alloc_pct2": 40.0,
    "alloc_pct3": 40.0,
}

def _v17_like(version: str, **overrides) -> StrategyFeatures:
    values = {
        "tp_mode": "v11_hybrid",
        "tp_offset_multiplier": 0.5,
        "long1_tp_uses_zone_offset": False,
        "short1_tp_uses_zone_offset": False,
        "reverse_breakeven_on_channel": True,
        "reverse_lock_bars": 136,
        "be_reentry_lock_bars": None,
        "deep_entry_overshoot_pct": 0.35,
        "deep_entry_reclaim": True,
        "mfe_zone_breakeven": True,
        "level0_mfe_zone_breakeven": True,
        "l1_reclaim_entry": False,
        "l1_knife_half_size": True,
        "no_progress_time_stop": True,
        "channel_mode": "baseline",
    }
    values.update(overrides)
    return StrategyFeatures(version=version, **values)


FEATURES_BY_VERSION = {
    "v18": _v17_like("v18"),
    "v19": _v17_like("v19", early_no_progress_time_stop=True),
    "v20": _v17_like("v20", l1_adverse_fail_exit=True),
    "v21": _v17_like("v21", short1_downtrend_filter=True),
    "v22": _v17_like("v22", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True),
    "v23": _v17_like("v23", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True),
    "v24": _v17_like("v24", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, breakout_early_fail_exit=True),
    "v25": _v17_like("v25", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, reverse_mfe_protect_during_lock=True),
    "v26": _v17_like("v26", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True),
    "v27": _v17_like("v27", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True, reverse_mfe_protect_during_lock=True),
    "v28": _v17_like("v28", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True, deep2_reclaim_fail_exit=True),
    "v29": _v17_like("v29", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True, breakout_no_progress_exit=True),
    "v30": _v17_like("v30", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True, deep2_reclaim_fail_exit=True, breakout_no_progress_exit=True),
}

VERSION_ORDER = ["v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30"]

