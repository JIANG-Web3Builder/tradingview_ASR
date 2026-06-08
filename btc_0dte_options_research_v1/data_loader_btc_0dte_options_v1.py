from __future__ import annotations

import pandas as pd

from config_btc_0dte_options_v1 import BTC_15M_FILE, END_DATE, START_DATE, V17_SIGNALS_FILE


def load_btc_15m() -> pd.DataFrame:
    if not BTC_15M_FILE.exists():
        raise FileNotFoundError(f"Missing BTC 15m file: {BTC_15M_FILE}")
    df = pd.read_csv(BTC_15M_FILE)
    if "timestamp" in df.columns and "open_time" not in df.columns:
        df = df.rename(columns={"timestamp": "open_time"})
    df["open_time"] = pd.to_datetime(df["open_time"], utc=False)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["bar_index"] = range(len(df))
    return df


def load_v17_signals() -> pd.DataFrame:
    if not V17_SIGNALS_FILE.exists():
        raise FileNotFoundError(f"Missing cached v17 signals: {V17_SIGNALS_FILE}")
    signals = pd.read_csv(V17_SIGNALS_FILE)
    signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=False)
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    signals = signals[(signals["entry_time"] >= start) & (signals["entry_time"] <= end)].copy()
    signals = signals.sort_values(["entry_time", "signal_id"]).reset_index(drop=True)
    return signals


def validate_btc_15m(df: pd.DataFrame) -> dict:
    spacing = df["open_time"].diff().dropna()
    bad_spacing = int((spacing != pd.Timedelta(minutes=15)).sum())
    return {
        "rows": int(len(df)),
        "start": df["open_time"].min().strftime("%Y-%m-%d %H:%M:%S") if len(df) else "",
        "end": df["open_time"].max().strftime("%Y-%m-%d %H:%M:%S") if len(df) else "",
        "bad_15m_spacing_count": bad_spacing,
    }
