from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(r"D:\workspace\20260325\asr_btc_multi_factor")
OUTPUT_DIR = BASE_DIR / "output_multi_factor_v1"
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
    l1_rsi_long_max: float | None = None
    l1_rsi_short_min: float | None = None
    deep_rsi_long_max: float | None = None
    deep_rsi_short_min: float | None = None
    l1_rsi_slope_confirm: bool = False
    l1_antirunaway_filter: bool = False
    l1_volume_confirm: bool = False
    high_volatility_filter: bool = False
    trend_entry_volume_filter: bool = False
    breakout_direction_filter: bool = False
    long1_alloc_multiplier: float = 1.0
    short1_alloc_multiplier: float = 1.0
    deep_alloc_multiplier: float = 1.0
    l1_mfe_protect_multiplier: float = 1.0

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


def _v26_base(version: str, **overrides) -> StrategyFeatures:
    values = {
        "early_no_progress_time_stop": True,
        "l1_adverse_fail_exit": True,
        "short1_downtrend_filter": True,
        "deep2_adverse_fail_exit": True,
        "breakout_early_fail_exit": True,
    }
    values.update(overrides)
    return _v17_like(version, **values)


FEATURES_BY_VERSION = {
    "v31": _v26_base("v31"),
    "v32": _v26_base("v32", l1_rsi_long_max=35.0, l1_rsi_short_min=65.0, deep_rsi_long_max=45.0, deep_rsi_short_min=55.0),
    "v33": _v26_base("v33", l1_rsi_slope_confirm=True),
    "v34": _v26_base("v34", l1_antirunaway_filter=True),
    "v35": _v26_base("v35", l1_volume_confirm=True),
    "v36": _v26_base("v36", high_volatility_filter=True),
    "v37": _v26_base("v37", trend_entry_volume_filter=True, breakout_direction_filter=True),
    "v38": _v26_base("v38", l1_rsi_long_max=38.0, l1_rsi_short_min=62.0, l1_rsi_slope_confirm=True, l1_antirunaway_filter=True),
    "v39": _v26_base("v39", l1_rsi_long_max=38.0, l1_rsi_short_min=62.0, l1_antirunaway_filter=True, l1_volume_confirm=True, high_volatility_filter=True),
    "v40": _v26_base("v40", l1_rsi_long_max=38.0, l1_rsi_short_min=62.0, deep_rsi_long_max=45.0, deep_rsi_short_min=55.0, l1_rsi_slope_confirm=True, l1_antirunaway_filter=True, l1_volume_confirm=True, high_volatility_filter=True, trend_entry_volume_filter=True, breakout_direction_filter=True),
    "v41": _v26_base("v41", long1_alloc_multiplier=0.5, short1_alloc_multiplier=0.5),
    "v42": _v26_base("v42", short1_alloc_multiplier=0.5),
    "v43": _v26_base("v43", long1_alloc_multiplier=0.5, short1_alloc_multiplier=0.5, deep_alloc_multiplier=1.125),
    "v44": _v26_base("v44", short1_alloc_multiplier=0.5, deep_alloc_multiplier=1.125),
    "v45": _v26_base("v45", l1_mfe_protect_multiplier=0.5),
    "v46": _v26_base("v46", l1_mfe_protect_multiplier=0.75),
    "v47": _v26_base("v47", long1_alloc_multiplier=0.5, short1_alloc_multiplier=0.5, l1_mfe_protect_multiplier=0.75),
    "v48": _v26_base("v48", short1_alloc_multiplier=0.5, l1_mfe_protect_multiplier=0.75),
    "v49": _v26_base("v49", long1_alloc_multiplier=0.5, short1_alloc_multiplier=0.5, deep_alloc_multiplier=1.125, l1_mfe_protect_multiplier=0.75),
    "v50": _v26_base("v50", short1_alloc_multiplier=0.5, deep_alloc_multiplier=1.125, l1_mfe_protect_multiplier=0.75),
}

VERSION_ORDER = ["v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40", "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v50"]


