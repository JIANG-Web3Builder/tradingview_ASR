from __future__ import annotations

import json
import sys

import pandas as pd

from config_polymarket_multi_horizon_v1 import CACHED_INDICATOR_FILE, CACHED_SIGNAL_META_FILE, CACHED_SIGNALS_FILE, CACHED_SOURCE_EQUITY_FILE, CHANNEL_RESEARCH_DIR, DATA_POLY_DIR, SIGNAL_VERSION


def _ensure_channel_import_path() -> None:
    channel_path = str(CHANNEL_RESEARCH_DIR)
    if channel_path not in sys.path:
        sys.path.insert(0, channel_path)


def _input_meta(price_data: pd.DataFrame) -> dict:
    close = price_data["close"].astype(float) if len(price_data) else pd.Series(dtype=float)
    return {
        "signal_version": SIGNAL_VERSION,
        "rows": int(len(price_data)),
        "start": pd.to_datetime(price_data["open_time"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S") if len(price_data) else "",
        "end": pd.to_datetime(price_data["open_time"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S") if len(price_data) else "",
        "first_close": round(float(close.iloc[0]), 8) if len(close) else None,
        "last_close": round(float(close.iloc[-1]), 8) if len(close) else None,
        "close_sum": round(float(close.sum()), 6) if len(close) else None,
    }


def _load_cached_outputs(price_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not (CACHED_SIGNALS_FILE.exists() and CACHED_INDICATOR_FILE.exists() and CACHED_SOURCE_EQUITY_FILE.exists() and CACHED_SIGNAL_META_FILE.exists()):
        return None
    meta = json.loads(CACHED_SIGNAL_META_FILE.read_text(encoding="utf-8"))
    if meta != _input_meta(price_data):
        return None

    signals = pd.read_csv(CACHED_SIGNALS_FILE)
    indicator_data = pd.read_csv(CACHED_INDICATOR_FILE)
    equity_df = pd.read_csv(CACHED_SOURCE_EQUITY_FILE)
    signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=False)
    indicator_data["open_time"] = pd.to_datetime(indicator_data["open_time"], utc=False)
    equity_df["open_time"] = pd.to_datetime(equity_df["open_time"], utc=False)
    return signals, indicator_data, equity_df


def _save_cached_outputs(price_data: pd.DataFrame, signals: pd.DataFrame, indicator_data: pd.DataFrame, equity_df: pd.DataFrame) -> None:
    DATA_POLY_DIR.mkdir(parents=True, exist_ok=True)
    signals.to_csv(CACHED_SIGNALS_FILE, index=False)
    indicator_data.to_csv(CACHED_INDICATOR_FILE, index=False)
    equity_df.to_csv(CACHED_SOURCE_EQUITY_FILE, index=False)
    CACHED_SIGNAL_META_FILE.write_text(json.dumps(_input_meta(price_data), indent=2), encoding="utf-8")


def build_v17_signals(price_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cached = _load_cached_outputs(price_data)
    if cached is not None:
        return cached

    _ensure_channel_import_path()
    from engine_channel_v4 import run_backtest
    from indicators_channel_v4 import compute_indicators
    from strategy_versions_channel_v4 import get_version_features

    features = get_version_features(SIGNAL_VERSION)
    indicator_data = compute_indicators(price_data.copy(), {"channel_mode": features.channel_mode})
    equity_df, trades_df = run_backtest(indicator_data.copy(), SIGNAL_VERSION)

    if trades_df.empty:
        return pd.DataFrame(), indicator_data, equity_df

    signal_cols = [
        "version",
        "trade_id",
        "side",
        "entry_id",
        "entry_time",
        "entry_price",
        "qty",
        "level_snapshot",
        "trend_mode_at_entry",
        "breakout_context",
        "reverse_context",
    ]
    signals = trades_df[signal_cols].copy()
    signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=False)

    price_lookup = indicator_data[[
        "open_time",
        "bar_index",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "rsiVal",
        "atr_",
        "atrPct",
        "smoothMid",
        "midHigh",
        "midLow",
        "smoothRes",
        "smoothSup",
        "zoneOffset",
        "channelWidth",
    ]].copy()
    price_lookup = price_lookup.rename(columns={
        "open_time": "entry_time",
        "bar_index": "signal_bar_index",
        "open": "signal_open",
        "high": "signal_high",
        "low": "signal_low",
        "close": "signal_close",
        "volume": "signal_volume",
    })

    signals = signals.merge(price_lookup, on="entry_time", how="left")
    signals = signals.dropna(subset=["signal_bar_index", "signal_close"]).copy()
    signals["signal_bar_index"] = signals["signal_bar_index"].astype(int)
    signals["signal_hour"] = signals["entry_time"].dt.hour
    signals["signal_month"] = signals["entry_time"].dt.to_period("M").astype(str)
    signals["candle_direction"] = "flat"
    signals.loc[signals["signal_close"] > signals["signal_open"], "candle_direction"] = "bullish"
    signals.loc[signals["signal_close"] < signals["signal_open"], "candle_direction"] = "bearish"
    signals["candle_body_pct"] = (signals["signal_close"] - signals["signal_open"]).abs() / signals["signal_close"] * 100.0
    signals["atr_pct_bucket"] = pd.qcut(signals["atrPct"], q=4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
    signals["rsi_bucket"] = pd.cut(signals["rsiVal"], bins=[0, 30, 40, 50, 60, 70, 100], include_lowest=True)
    signals = signals.sort_values(["entry_time", "trade_id"]).reset_index(drop=True)
    signals.insert(0, "signal_id", range(1, len(signals) + 1))
    _save_cached_outputs(price_data, signals, indicator_data, equity_df)
    return signals, indicator_data, equity_df
