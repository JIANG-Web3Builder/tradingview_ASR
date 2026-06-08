from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\workspace\20260325\asr_eth_channel_research_v1")
SOURCE_DIR = Path(r"D:\workspace\20260325\asr_btc_channel_research_v1")
DATA_DIR = Path(r"D:\workspace\20260325\data")
OUTPUT_DIR = BASE_DIR / "output_channel_eth_v2"
DATA_FILE_15M = DATA_DIR / "ETHUSDT_15m.csv"
START_DATE = "2024-01-01 00:00:00"
END_DATE = "2026-05-01 23:59:59"
INITIAL_CAPITAL = 10000.0
SYMBOL = "ETHUSDT"

if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import config_channel_v5 as base_config
from engine_channel_v5 import run_backtest
from indicators_channel_v5 import pine_atr, pine_rsi
from metrics_channel_v5 import compute_summary, save_summary

REQUIRED_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
VERSION_ORDER = [
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
    "v17",
    "v18",
    "v19",
    "v20",
    "v21",
    "v22",
    "v23",
    "v24",
    "v25",
    "v26",
    "v27",
    "v28",
    "v29",
    "v30",
]

ETH_CHANNEL_PARAMS = {
    "channel_mode": "eth_vol_scaled",
    "eth_target_atr_pct": 0.00335,
    "eth_asset_scale_min": 1.0,
    "eth_asset_scale_max": 1.8,
    "eth_regime_min": 0.75,
    "eth_regime_max": 1.75,
    "eth_regime_exponent": 0.5,
}


def _feature(version: str, **overrides) -> base_config.StrategyFeatures:
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
        "channel_mode": ETH_CHANNEL_PARAMS["channel_mode"],
        "early_no_progress_time_stop": False,
        "l1_adverse_fail_exit": False,
        "short1_downtrend_filter": False,
        "deep2_adverse_fail_exit": False,
        "breakout_early_fail_exit": False,
        "reverse_mfe_protect_during_lock": False,
        "deep2_reclaim_fail_exit": False,
        "breakout_no_progress_exit": False,
    }
    values.update(overrides)
    return base_config.StrategyFeatures(version=version, **values)


def build_features_by_version() -> dict[str, base_config.StrategyFeatures]:
    return {
        "v10": _feature(
            "v10",
            tp_mode="v9_unified",
            tp_offset_multiplier=0.6,
            be_reentry_lock_bars=136,
            deep_entry_overshoot_pct=None,
            deep_entry_reclaim=False,
            mfe_zone_breakeven=False,
            level0_mfe_zone_breakeven=False,
            l1_knife_half_size=False,
            no_progress_time_stop=False,
        ),
        "v11": _feature(
            "v11",
            deep_entry_overshoot_pct=None,
            deep_entry_reclaim=False,
            mfe_zone_breakeven=False,
            level0_mfe_zone_breakeven=False,
            l1_knife_half_size=False,
            no_progress_time_stop=False,
        ),
        "v12": _feature(
            "v12",
            mfe_zone_breakeven=False,
            level0_mfe_zone_breakeven=False,
            l1_knife_half_size=False,
            no_progress_time_stop=False,
        ),
        "v13": _feature(
            "v13",
            level0_mfe_zone_breakeven=False,
            l1_knife_half_size=False,
            no_progress_time_stop=False,
        ),
        "v14": _feature(
            "v14",
            l1_knife_half_size=False,
            no_progress_time_stop=False,
        ),
        "v15": _feature(
            "v15",
            l1_reclaim_entry=True,
            l1_knife_half_size=False,
            no_progress_time_stop=False,
        ),
        "v16": _feature(
            "v16",
            no_progress_time_stop=False,
        ),
        "v17": _feature("v17"),
        "v18": _feature("v18"),
        "v19": _feature("v19", early_no_progress_time_stop=True),
        "v20": _feature("v20", l1_adverse_fail_exit=True),
        "v21": _feature("v21", short1_downtrend_filter=True),
        "v22": _feature("v22", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True),
        "v23": _feature("v23", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True),
        "v24": _feature("v24", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, breakout_early_fail_exit=True),
        "v25": _feature("v25", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, reverse_mfe_protect_during_lock=True),
        "v26": _feature("v26", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True),
        "v27": _feature("v27", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True, reverse_mfe_protect_during_lock=True),
        "v28": _feature("v28", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True, deep2_reclaim_fail_exit=True),
        "v29": _feature("v29", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True, breakout_no_progress_exit=True),
        "v30": _feature("v30", early_no_progress_time_stop=True, l1_adverse_fail_exit=True, short1_downtrend_filter=True, deep2_adverse_fail_exit=True, breakout_early_fail_exit=True, deep2_reclaim_fail_exit=True, breakout_no_progress_exit=True),
    }


def configure_feature_registry() -> None:
    features = build_features_by_version()
    base_config.FEATURES_BY_VERSION.clear()
    base_config.FEATURES_BY_VERSION.update(features)


def load_15m_data() -> pd.DataFrame:
    if not DATA_FILE_15M.exists():
        raise FileNotFoundError(
            f"Missing ETH 15m data at {DATA_FILE_15M}. Run d:/workspace/20260325/data/download_eth_v1.py first."
        )
    df = pd.read_csv(DATA_FILE_15M)
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
    return df


def validate_15m_data(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError("15m data is empty after filtering")
    spacing = df["open_time"].diff().dropna()
    expected = pd.Timedelta(minutes=15)
    bad_spacing = int((spacing != expected).sum())
    return {
        "rows": int(len(df)),
        "start": df["open_time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
        "end": df["open_time"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "bad_spacing_count": bad_spacing,
    }


def compute_indicators_eth_v2(df: pd.DataFrame, base_params: dict | None = None) -> pd.DataFrame:
    params = dict(base_config.BASE_PARAMS)
    params.update(ETH_CHANNEL_PARAMS)
    if base_params:
        params.update(base_params)

    out = df.copy()
    current_tf = base_config.TIMEFRAME_MINUTES
    tf_multiplier = params["base_tf_minutes"] / current_tf

    hl_length = max(1, round(params["hl_length_input"] * tf_multiplier))
    atr_length = max(1, round(params["atr_length_input"] * tf_multiplier))
    vov_length = max(1, round(params["vov_length_input"] * tf_multiplier))
    smooth_factor = max(1, round(params["smooth_factor_input"] * tf_multiplier))
    time_stop_bars = max(1, round(params["time_stop_bars_in"] * tf_multiplier))
    trend_ma = max(10, round(params["trend_ma_input"] * tf_multiplier))
    cooldown_bars = max(0, round(params["cooldown_bars_in"] * tf_multiplier))

    out["hl2"] = (out["high"] + out["low"]) / 2.0
    out["typicalPrice"] = (out["high"] + out["low"] + out["close"]) / 3.0
    rolling_volume = out["volume"].rolling(hl_length, min_periods=hl_length).sum()
    rolling_vp = (out["typicalPrice"] * out["volume"]).rolling(hl_length, min_periods=hl_length).sum()
    out["volumeWeightedMidLine"] = rolling_vp / rolling_volume.replace(0.0, np.nan)
    out["midLine"] = out["hl2"].rolling(hl_length, min_periods=hl_length).mean()
    out["tr_"], out["atr_"] = pine_atr(out["high"], out["low"], out["close"], atr_length)
    out["roc_close"] = out["close"].pct_change(1) * 100.0
    out["rocVol"] = out["roc_close"].abs().rolling(20, min_periods=20).std() / out["close"] * 10000.0
    out["compVol"] = out["tr_"] * 0.4 + out["atr_"] * 0.4 + out["rocVol"] * 0.2
    out["vovDenom"] = out["compVol"].rolling(vov_length, min_periods=vov_length).mean()
    out["vov"] = out["compVol"].rolling(vov_length, min_periods=vov_length).std() / out["vovDenom"]
    out["vovSafe"] = out["vov"].where((~out["vov"].isna()) & (out["vov"] <= 10.0), 0.0)

    base_offset = out["midLine"] * params["base_width_pct"] / 100.0 / 2.0
    out["atrPct"] = out["atr_"] / out["close"]
    out["atrPctBase"] = out["atrPct"].rolling(vov_length, min_periods=vov_length).median()
    fallback_window = max(20, atr_length)
    out["atrPctBase"] = out["atrPctBase"].fillna(out["atrPct"].rolling(fallback_window, min_periods=fallback_window).median())
    out["atrPctBase"] = out["atrPctBase"].fillna(params["eth_target_atr_pct"])
    out["atrRegime"] = (out["atrPct"] / out["atrPctBase"]).replace([np.inf, -np.inf], np.nan)
    out["atrRegimeSafe"] = out["atrRegime"].clip(lower=params["eth_regime_min"], upper=params["eth_regime_max"]).fillna(1.0)
    out["assetWidthScale"] = (out["atrPctBase"] / params["eth_target_atr_pct"]).clip(lower=params["eth_asset_scale_min"], upper=params["eth_asset_scale_max"]).fillna(1.0)
    out["dynamicOffset"] = base_offset * out["assetWidthScale"] * (out["atrRegimeSafe"] ** params["eth_regime_exponent"]) * (1.0 + out["vovSafe"] * params["adjust_factor"])
    out["smoothRes"] = (out["midLine"] + out["dynamicOffset"]).ewm(span=smooth_factor, adjust=False, min_periods=smooth_factor).mean()
    out["smoothSup"] = (out["midLine"] - out["dynamicOffset"]).ewm(span=smooth_factor, adjust=False, min_periods=smooth_factor).mean()
    out["smoothMid"] = out["midLine"].ewm(span=smooth_factor, adjust=False, min_periods=smooth_factor).mean()
    out["swingHigh"] = np.nan
    out["swingLow"] = np.nan
    out["volumeSmoothMid"] = np.nan
    out["internalMid"] = out["smoothMid"]
    out["midHigh"] = (out["internalMid"] + out["smoothRes"]) / 2.0
    out["midLow"] = (out["internalMid"] + out["smoothSup"]) / 2.0
    out["superbuy"] = out["smoothMid"] + out["dynamicOffset"] * 1.618
    out["supersell"] = out["smoothMid"] - out["dynamicOffset"] * 1.618
    out["uperLine"] = out["smoothRes"] + out["dynamicOffset"] * 0.25
    out["downerLine"] = out["smoothSup"] - out["dynamicOffset"] * 0.25

    diff = (out["smoothRes"] - out["smoothSup"]).abs().fillna(base_config.MINTICK * 2.0)
    out["channelWidth"] = np.maximum(diff, base_config.MINTICK * 2.0)
    out["zoneOffset"] = np.maximum(base_config.MINTICK, out["channelWidth"] * params["zone_channel_pct"])
    out["maLine"] = out["close"].rolling(trend_ma, min_periods=trend_ma).mean()
    out["rsiVal"] = pine_rsi(out["close"], params["rsi_period"])
    out["uptrend"] = out["close"] > out["maLine"]
    out["downtrend"] = out["close"] < out["maLine"]
    out["reverseLongStopRef"] = out["high"].rolling(params["reverse_stop_lookback"], min_periods=params["reverse_stop_lookback"]).max().shift(1)
    out["reverseShortStopRef"] = out["low"].rolling(params["reverse_stop_lookback"], min_periods=params["reverse_stop_lookback"]).min().shift(1)
    out["reverseStopBuffer"] = np.maximum(base_config.MINTICK, out["atr_"] * 0.1)
    out["reverseShortStopPxRef"] = np.where(
        out["reverseLongStopRef"].isna(),
        np.nan,
        np.maximum(out["high"], out["reverseLongStopRef"]) + out["reverseStopBuffer"],
    )
    out["reverseLongStopPxRef"] = np.where(
        out["reverseShortStopRef"].isna(),
        np.nan,
        np.minimum(out["low"], out["reverseShortStopRef"]) - out["reverseStopBuffer"],
    )
    out["tfMultiplier"] = tf_multiplier
    out["hlLength"] = hl_length
    out["atrLength"] = atr_length
    out["vovLength"] = vov_length
    out["smoothFactor"] = smooth_factor
    out["timeStopBars"] = time_stop_bars
    out["trendMA"] = trend_ma
    out["cooldownBars"] = cooldown_bars
    return out


def plot_equity_curve(equity_df: pd.DataFrame, output_path: Path, title: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_df["open_time"], equity_df["equity"], linewidth=1.4)
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_equity_comparison(curves: dict[str, pd.DataFrame], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for version, df in curves.items():
        ax.plot(df["open_time"], df["equity"], linewidth=1.1, label=version)
    ax.set_title("ASR ETH Equity Curve Comparison (v10-v30, ETH Vol Scaled)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend(ncols=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_feature_registry()
    data = load_15m_data()
    validation = validate_15m_data(data)
    (OUTPUT_DIR / "data_validation_channel_eth_v2.json").write_text(pd.Series(validation).to_json(indent=2), encoding="utf-8")

    comparison_curves: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for version in VERSION_ORDER:
        features = base_config.FEATURES_BY_VERSION[version]
        version_dir = OUTPUT_DIR / version
        version_dir.mkdir(parents=True, exist_ok=True)

        indicator_data = compute_indicators_eth_v2(data, {"channel_mode": features.channel_mode})
        equity_df, trades_df = run_backtest(indicator_data.copy(), version)
        summary = compute_summary(equity_df, trades_df, INITIAL_CAPITAL)
        summary["version"] = version
        summary["symbol"] = SYMBOL
        summary["channel_mode"] = features.channel_mode
        summary["data_start"] = validation["start"]
        summary["data_end"] = validation["end"]
        summary["data_bad_spacing_count"] = validation["bad_spacing_count"]
        summary["avg_channel_pct"] = float((indicator_data["channelWidth"] / indicator_data["close"]).dropna().mean())
        summary["avg_atr_pct"] = float((indicator_data["atr_"] / indicator_data["close"]).dropna().mean())
        summary["avg_channel_atr_ratio"] = float((indicator_data["channelWidth"] / indicator_data["atr_"]).replace([np.inf, -np.inf], np.nan).dropna().mean())
        summary["avg_asset_width_scale"] = float(indicator_data["assetWidthScale"].dropna().mean())
        summary_rows.append(summary)

        equity_df.to_csv(version_dir / "equity_curve_channel_eth_v2.csv", index=False)
        daily_equity_df = equity_df.copy()
        daily_equity_df["date"] = pd.to_datetime(daily_equity_df["open_time"]).dt.date
        daily_equity_df = daily_equity_df.groupby("date", as_index=False).tail(1)
        daily_equity_df.to_csv(version_dir / "daily_equity_channel_eth_v2.csv", index=False)
        trades_df.to_csv(version_dir / "trades_channel_eth_v2.csv", index=False)
        save_summary(version_dir / "summary_channel_eth_v2.json", summary)

        subtitle = (
            f"Return {summary['total_return']:.2%} | Sharpe {summary['sharpe']:.3f} | "
            f"MaxDD {summary['max_drawdown']:.2%} | Trades {summary['trade_count']} | Channel {features.channel_mode}"
        )
        plot_equity_curve(equity_df, version_dir / "equity_curve_channel_eth_v2.png", f"ASR ETH {version}", subtitle)
        comparison_curves[version] = equity_df

    plot_equity_comparison(comparison_curves, OUTPUT_DIR / "equity_comparison_channel_eth_v2.png")
    pd.DataFrame(summary_rows).sort_values("version").to_csv(OUTPUT_DIR / "summary_comparison_channel_eth_v2.csv", index=False)


if __name__ == "__main__":
    main()
