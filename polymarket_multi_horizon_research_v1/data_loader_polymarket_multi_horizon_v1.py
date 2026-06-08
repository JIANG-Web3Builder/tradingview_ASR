from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config_polymarket_multi_horizon_v1 import CACHED_DATA_FILE_15M, CACHED_DATA_META_FILE_15M, DATA_FILE_15M, DATA_POLY_DIR, END_DATE, START_DATE, TIMEFRAME_MINUTES

REQUIRED_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]


def _source_meta(path: Path) -> dict:
    stat = path.stat()
    return {
        "source_path": str(path),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "start_date": START_DATE,
        "end_date": END_DATE,
    }


def _cache_is_valid(source_path: Path) -> bool:
    if not (CACHED_DATA_FILE_15M.exists() and CACHED_DATA_META_FILE_15M.exists()):
        return False
    meta = json.loads(CACHED_DATA_META_FILE_15M.read_text(encoding="utf-8"))
    return meta.get("source") == _source_meta(source_path)


def load_15m_data(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or DATA_FILE_15M
    if csv_path is None and _cache_is_valid(path):
        df = pd.read_csv(CACHED_DATA_FILE_15M)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=False)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["bar_index"] = df["bar_index"].astype(int)
        return df

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], utc=False)
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    df = df[(df["open_time"] >= pd.Timestamp(START_DATE)) & (df["open_time"] <= pd.Timestamp(END_DATE))].reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["bar_index"] = range(len(df))
    if csv_path is None:
        DATA_POLY_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHED_DATA_FILE_15M, index=False)
        CACHED_DATA_META_FILE_15M.write_text(json.dumps({"source": _source_meta(path), "validation": validate_15m_data(df)}, indent=2), encoding="utf-8")
    return df


def validate_15m_data(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("15m data is empty after filtering")

    spacing = df["open_time"].diff().dropna()
    expected = pd.Timedelta(minutes=TIMEFRAME_MINUTES)
    bad_spacing = int((spacing != expected).sum())

    return {
        "rows": int(len(df)),
        "start": df["open_time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "end": df["open_time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "bad_spacing_count": bad_spacing,
    }
