from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config_polymarket_multi_horizon_v1 import INITIAL_BANKROLL, STRATEGY_VARIANTS, TRAIN_END
from market_simulator_polymarket_multi_horizon_v1 import build_equity_curve


def apply_variant_filter(df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    variant = next(item for item in STRATEGY_VARIANTS if item.name == variant_name)
    out = df.copy()
    if variant.sides is not None:
        out = out[out["side"].isin(variant.sides)]
    if variant.entry_ids is not None:
        out = out[out["entry_id"].isin(variant.entry_ids)]
    if variant.max_horizon_bars is not None:
        out = out[out["horizon_bars"] <= variant.max_horizon_bars]
    if variant.min_horizon_bars is not None:
        out = out[out["horizon_bars"] >= variant.min_horizon_bars]
    return out.copy()


def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    train_end = pd.Timestamp(TRAIN_END)
    out["sample_split"] = np.where(pd.to_datetime(out["entry_time"]) < train_end, "IS", "OOS")
    return out


def add_all_split(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    with_split = add_split_column(df)
    all_split = with_split.copy()
    all_split["sample_split"] = "ALL"
    return pd.concat([all_split, with_split], ignore_index=True, sort=False)


def max_drawdown_from_pnls(pnls: pd.Series, initial_bankroll: float = INITIAL_BANKROLL) -> float:
    if pnls.empty:
        return np.nan
    equity = initial_bankroll + pnls.cumsum()
    roll_max = equity.cummax()
    return float((equity / roll_max - 1.0).min())


def longest_streak(values: pd.Series, target: bool) -> int:
    best = 0
    current = 0
    for value in values.astype(bool):
        if value == target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def summarize_priced_bets(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if "concurrency_mode" not in df.columns:
        df = df.copy()
        df["concurrency_mode"] = "independent"
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ordered = group.sort_values(["settlement_time", "signal_id"])
        wins = ordered["won"].astype(bool)
        gross_profit = ordered.loc[ordered["pnl"] > 0, "pnl"].sum()
        gross_loss = -ordered.loc[ordered["pnl"] < 0, "pnl"].sum()
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update({
            "bet_count": int(len(ordered)),
            "win_count": int(wins.sum()),
            "loss_count": int((~wins).sum()),
            "win_rate": float(wins.mean()) if len(ordered) else np.nan,
            "avg_direction_return_pct": float(ordered["direction_return_pct"].mean()),
            "median_direction_return_pct": float(ordered["direction_return_pct"].median()),
            "avg_stake": float(ordered["stake"].mean()),
            "total_stake": float(ordered["stake"].sum()),
            "total_pnl": float(ordered["pnl"].sum()),
            "roi_on_stake": float(ordered["pnl"].sum() / ordered["stake"].sum()) if ordered["stake"].sum() else np.nan,
            "avg_pnl": float(ordered["pnl"].mean()),
            "final_equity": float(INITIAL_BANKROLL + ordered["pnl"].sum()),
            "max_drawdown": max_drawdown_from_pnls(ordered["pnl"]),
            "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
            "longest_win_streak": longest_streak(wins, True),
            "longest_loss_streak": longest_streak(wins, False),
            "breakeven_buy_price_before_fees": float(wins.mean()) if len(ordered) else np.nan,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def build_accuracy_by_group(binary_bets: pd.DataFrame) -> pd.DataFrame:
    if binary_bets.empty:
        return pd.DataFrame()
    group_sets = [
        ["horizon_minutes"],
        ["side", "horizon_minutes"],
        ["entry_id", "horizon_minutes"],
        ["side", "entry_id", "horizon_minutes"],
        ["signal_hour", "horizon_minutes"],
        ["atr_pct_bucket", "horizon_minutes"],
        ["rsi_bucket", "horizon_minutes"],
        ["candle_direction", "horizon_minutes"],
    ]
    rows = []
    for group_cols in group_sets:
        groupby_key = group_cols[0] if len(group_cols) == 1 else group_cols
        for keys, group in binary_bets.groupby(groupby_key, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: key for col, key in zip(group_cols, keys)}
            row["group_by"] = "|".join(group_cols)
            row["sample_count"] = int(len(group))
            row["win_rate"] = float(group["won"].astype(bool).mean())
            row["avg_direction_return_pct"] = float(group["direction_return_pct"].mean())
            row["median_direction_return_pct"] = float(group["direction_return_pct"].median())
            rows.append(row)
    return pd.DataFrame(rows)


def build_variant_summary(priced_bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    priced_bets = add_all_split(priced_bets)
    for variant in STRATEGY_VARIANTS:
        filtered = apply_variant_filter(priced_bets, variant.name)
        if filtered.empty:
            continue
        summary = summarize_priced_bets(
            filtered,
            ["concurrency_mode", "sample_split", "horizon_minutes", "buy_price", "fee_drag"],
        )
        summary.insert(0, "variant", variant.name)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_price_sensitivity(priced_bets: pd.DataFrame) -> pd.DataFrame:
    priced_bets = add_all_split(priced_bets)
    rows = []
    for variant in STRATEGY_VARIANTS:
        filtered = apply_variant_filter(priced_bets, variant.name)
        if filtered.empty:
            continue
        summary = summarize_priced_bets(
            filtered,
            ["concurrency_mode", "sample_split", "horizon_minutes", "buy_price", "fee_drag"],
        )
        summary.insert(0, "variant", variant.name)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def save_summary_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_key_equity_curves(priced_bets: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    key_cases = [
        ("all_signals", 60, 0.50, 0.0),
        ("all_signals", 360, 0.50, 0.0),
        ("all_signals", 1440, 0.50, 0.0),
        ("long_only", 60, 0.50, 0.0),
        ("long_only", 360, 0.50, 0.0),
        ("long_only", 1440, 0.50, 0.0),
    ]
    for variant_name, horizon_minutes, buy_price, fee_drag in key_cases:
        filtered = apply_variant_filter(priced_bets, variant_name)
        filtered = filtered[
            (filtered["horizon_minutes"] == horizon_minutes)
            & (filtered["buy_price"] == buy_price)
            & (filtered["fee_drag"] == fee_drag)
            & (filtered["concurrency_mode"] == "independent")
        ].copy()
        if filtered.empty:
            continue
        curve = build_equity_curve(filtered)
        path = output_dir / f"equity_curve_{variant_name}_{horizon_minutes}m_price_{str(buy_price).replace('.', 'p')}_fee_{str(fee_drag).replace('.', 'p')}_v1.csv"
        curve.to_csv(path, index=False)
        paths.append(path)
    return paths
