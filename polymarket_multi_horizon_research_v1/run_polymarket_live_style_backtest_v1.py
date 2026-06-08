from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_polymarket_multi_horizon_v1 import INITIAL_BANKROLL, OUTPUT_DIR, TRAIN_END
from data_loader_polymarket_multi_horizon_v1 import load_15m_data, validate_15m_data
from market_simulator_polymarket_multi_horizon_v1 import build_binary_bets
from signal_builder_polymarket_multi_horizon_v1 import build_v17_signals

LIVE_OUTPUT_DIR = OUTPUT_DIR / "live_style_v1"
RAW_BETS_FILE = OUTPUT_DIR / "binary_bets_raw_multi_horizon_v1.csv"
REPORT_FILE = LIVE_OUTPUT_DIR / "live_style_report_v1.md"
TRADES_FILE = LIVE_OUTPUT_DIR / "live_style_trades_v1.csv"
SUMMARY_FILE = LIVE_OUTPUT_DIR / "live_style_summary_v1.csv"
EQUITY_FILE = LIVE_OUTPUT_DIR / "live_style_equity_curves_v1.csv"
JSON_FILE = LIVE_OUTPUT_DIR / "live_style_summary_v1.json"
PLOT_FILE = LIVE_OUTPUT_DIR / "live_style_equity_comparison_v1.png"
CALIBRATION_FILE = LIVE_OUTPUT_DIR / "live_style_policy_calibration_v1.csv"


@dataclass(frozen=True)
class LivePolicy:
    name: str
    horizon_minutes: int
    side: str | None
    entry_ids: tuple[str, ...] | None
    market_buy_price: float
    fee_drag: float
    min_edge: float
    min_history: int
    lookback_days: int
    group_cols: tuple[str, ...]
    fallback_group_cols: tuple[str, ...]
    max_open_positions: int
    max_open_same_side: int
    stake_fraction: float
    max_stake_fraction: float
    kelly_fraction: float
    min_stake: float
    max_stake: float
    stop_trading_drawdown: float


POLICIES = [
    LivePolicy(
        name="long_360_edge_price_52",
        horizon_minutes=360,
        side="long",
        entry_ids=None,
        market_buy_price=0.52,
        fee_drag=0.03,
        min_edge=0.03,
        min_history=40,
        lookback_days=365,
        group_cols=("side", "entry_id", "horizon_minutes"),
        fallback_group_cols=("side", "horizon_minutes"),
        max_open_positions=1,
        max_open_same_side=1,
        stake_fraction=0.01,
        max_stake_fraction=0.02,
        kelly_fraction=0.25,
        min_stake=25.0,
        max_stake=200.0,
        stop_trading_drawdown=-0.15,
    ),
    LivePolicy(
        name="long_360_strict_price_54",
        horizon_minutes=360,
        side="long",
        entry_ids=None,
        market_buy_price=0.54,
        fee_drag=0.04,
        min_edge=0.04,
        min_history=60,
        lookback_days=365,
        group_cols=("side", "entry_id", "horizon_minutes"),
        fallback_group_cols=("side", "horizon_minutes"),
        max_open_positions=1,
        max_open_same_side=1,
        stake_fraction=0.008,
        max_stake_fraction=0.015,
        kelly_fraction=0.20,
        min_stake=25.0,
        max_stake=150.0,
        stop_trading_drawdown=-0.12,
    ),
    LivePolicy(
        name="deep_720_edge_price_52",
        horizon_minutes=720,
        side=None,
        entry_ids=("Long2", "Long3", "Short2", "Short3"),
        market_buy_price=0.52,
        fee_drag=0.03,
        min_edge=0.05,
        min_history=25,
        lookback_days=540,
        group_cols=("entry_id", "horizon_minutes"),
        fallback_group_cols=("horizon_minutes",),
        max_open_positions=1,
        max_open_same_side=1,
        stake_fraction=0.006,
        max_stake_fraction=0.012,
        kelly_fraction=0.15,
        min_stake=20.0,
        max_stake=120.0,
        stop_trading_drawdown=-0.10,
    ),
    LivePolicy(
        name="all_60_liquid_price_51",
        horizon_minutes=60,
        side=None,
        entry_ids=None,
        market_buy_price=0.51,
        fee_drag=0.025,
        min_edge=0.025,
        min_history=80,
        lookback_days=365,
        group_cols=("side", "entry_id", "horizon_minutes"),
        fallback_group_cols=("side", "horizon_minutes"),
        max_open_positions=1,
        max_open_same_side=1,
        stake_fraction=0.01,
        max_stake_fraction=0.02,
        kelly_fraction=0.20,
        min_stake=25.0,
        max_stake=200.0,
        stop_trading_drawdown=-0.15,
    ),
    LivePolicy(
        name="all_360_edge_price_52",
        horizon_minutes=360,
        side=None,
        entry_ids=None,
        market_buy_price=0.52,
        fee_drag=0.03,
        min_edge=0.035,
        min_history=50,
        lookback_days=365,
        group_cols=("side", "entry_id", "horizon_minutes"),
        fallback_group_cols=("side", "horizon_minutes"),
        max_open_positions=1,
        max_open_same_side=1,
        stake_fraction=0.008,
        max_stake_fraction=0.015,
        kelly_fraction=0.20,
        min_stake=25.0,
        max_stake=150.0,
        stop_trading_drawdown=-0.15,
    ),
]


def ensure_raw_binary_bets() -> pd.DataFrame:
    if RAW_BETS_FILE.exists():
        df = pd.read_csv(RAW_BETS_FILE)
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=False)
        df["settlement_time"] = pd.to_datetime(df["settlement_time"], utc=False)
        return df
    prices = load_15m_data()
    signals, indicator_data, _ = build_v17_signals(prices)
    df = build_binary_bets(signals, indicator_data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_BETS_FILE, index=False)
    return df


def policy_filter(df: pd.DataFrame, policy: LivePolicy) -> pd.DataFrame:
    out = df[df["horizon_minutes"] == policy.horizon_minutes].copy()
    if policy.side is not None:
        out = out[out["side"] == policy.side].copy()
    if policy.entry_ids is not None:
        out = out[out["entry_id"].isin(policy.entry_ids)].copy()
    return out.sort_values(["entry_time", "signal_id"]).reset_index(drop=True)


def historical_group(data: pd.DataFrame, row: pd.Series, cols: tuple[str, ...]) -> pd.DataFrame:
    if not cols:
        return data
    mask = pd.Series(True, index=data.index)
    for col in cols:
        mask &= data[col] == row[col]
    return data[mask]


def estimate_probability(history: pd.DataFrame, row: pd.Series, policy: LivePolicy) -> tuple[float, int, str]:
    end_time = pd.to_datetime(row["entry_time"])
    start_time = end_time - pd.Timedelta(days=policy.lookback_days)
    past = history[
        (pd.to_datetime(history["settlement_time"]) < end_time)
        & (pd.to_datetime(history["entry_time"]) >= start_time)
    ].copy()
    primary = historical_group(past, row, policy.group_cols)
    if len(primary) >= policy.min_history:
        return float(primary["won"].astype(bool).mean()), int(len(primary)), "primary"
    fallback = historical_group(past, row, policy.fallback_group_cols)
    if len(fallback) >= policy.min_history:
        return float(fallback["won"].astype(bool).mean()), int(len(fallback)), "fallback"
    return np.nan, int(max(len(primary), len(fallback))), "insufficient"


def settle_positions(open_positions: list[dict], now: pd.Timestamp, bankroll: float) -> tuple[list[dict], float, list[dict]]:
    remaining: list[dict] = []
    settled: list[dict] = []
    for position in open_positions:
        if pd.to_datetime(position["settlement_time"]) <= now:
            bankroll += float(position["pnl"])
            settled.append(position)
        else:
            remaining.append(position)
    return remaining, bankroll, settled


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
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


def simulate_policy(binary_bets: pd.DataFrame, policy: LivePolicy) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = policy_filter(binary_bets, policy)
    train_end = pd.Timestamp(TRAIN_END)
    cost_per_share = policy.market_buy_price * (1.0 + policy.fee_drag)
    open_positions: list[dict] = []
    trades: list[dict] = []
    calibration_rows: list[dict] = []
    bankroll = INITIAL_BANKROLL
    peak_equity = INITIAL_BANKROLL
    stopped = False

    for _, row in candidates.iterrows():
        entry_time = pd.to_datetime(row["entry_time"])
        open_positions, bankroll, _ = settle_positions(open_positions, entry_time, bankroll)
        peak_equity = max(peak_equity, bankroll)
        current_drawdown = bankroll / peak_equity - 1.0 if peak_equity else 0.0
        if current_drawdown <= policy.stop_trading_drawdown:
            stopped = True
        p_hat, history_count, estimator = estimate_probability(candidates, row, policy)
        edge = p_hat - cost_per_share if not pd.isna(p_hat) else np.nan
        is_oos = entry_time >= train_end
        skip_reason = ""
        if stopped:
            skip_reason = "drawdown_stop"
        elif not is_oos:
            skip_reason = "is_warmup"
        elif estimator == "insufficient":
            skip_reason = "insufficient_history"
        elif edge < policy.min_edge:
            skip_reason = "edge_too_small"
        elif len(open_positions) >= policy.max_open_positions:
            skip_reason = "max_open_positions"
        elif sum(1 for pos in open_positions if pos["side"] == row["side"]) >= policy.max_open_same_side:
            skip_reason = "max_open_same_side"

        calibration_rows.append({
            "policy": policy.name,
            "signal_id": int(row["signal_id"]),
            "entry_time": entry_time,
            "side": row["side"],
            "entry_id": row["entry_id"],
            "horizon_minutes": int(row["horizon_minutes"]),
            "p_hat": p_hat,
            "history_count": history_count,
            "estimator": estimator,
            "cost_per_share": cost_per_share,
            "edge": edge,
            "is_oos": bool(is_oos),
            "skip_reason": skip_reason,
        })

        if skip_reason:
            continue

        b = (1.0 - cost_per_share) / cost_per_share
        kelly_full = (b * p_hat - (1.0 - p_hat)) / b if b > 0 else 0.0
        kelly_fraction = max(0.0, min(kelly_full * policy.kelly_fraction, policy.max_stake_fraction))
        stake_fraction = min(policy.stake_fraction, kelly_fraction) if kelly_fraction > 0 else 0.0
        stake = bankroll * stake_fraction
        stake = min(policy.max_stake, max(policy.min_stake, stake))
        stake = min(stake, bankroll * policy.max_stake_fraction)
        if stake <= 0 or stake > bankroll:
            continue

        shares = stake / cost_per_share
        gross_payout = shares if bool(row["won"]) else 0.0
        pnl = gross_payout - stake
        trade = {
            "policy": policy.name,
            "signal_id": int(row["signal_id"]),
            "trade_id": int(row["trade_id"]),
            "entry_time": entry_time,
            "settlement_time": pd.to_datetime(row["settlement_time"]),
            "side": row["side"],
            "entry_id": row["entry_id"],
            "horizon_minutes": int(row["horizon_minutes"]),
            "signal_close": float(row["signal_close"]),
            "settlement_close": float(row["settlement_close"]),
            "direction_return_pct": float(row["direction_return_pct"]),
            "won": bool(row["won"]),
            "market_buy_price": policy.market_buy_price,
            "fee_drag": policy.fee_drag,
            "cost_per_share": cost_per_share,
            "p_hat": p_hat,
            "history_count": history_count,
            "estimator": estimator,
            "edge": edge,
            "stake": stake,
            "shares": shares,
            "gross_payout": gross_payout,
            "pnl": pnl,
            "return_on_stake": pnl / stake,
            "bankroll_at_entry": bankroll,
            "open_positions_at_entry": len(open_positions),
        }
        open_positions.append(trade)
        trades.append(trade)

    open_positions, bankroll, _ = settle_positions(open_positions, pd.Timestamp.max.tz_localize(None), bankroll)
    return pd.DataFrame(trades), pd.DataFrame(calibration_rows)


def build_equity(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["policy", "settlement_time", "pnl", "equity", "drawdown"])
    rows = []
    for policy, group in trades.groupby("policy", dropna=False):
        curve = group.sort_values(["settlement_time", "signal_id"]).groupby("settlement_time", as_index=False).agg(
            pnl=("pnl", "sum"),
            bet_count=("signal_id", "count"),
        )
        curve["policy"] = policy
        curve["equity"] = INITIAL_BANKROLL + curve["pnl"].cumsum()
        curve["roll_max"] = curve["equity"].cummax()
        curve["drawdown"] = curve["equity"] / curve["roll_max"] - 1.0
        rows.append(curve)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def summarize_live_trades(trades: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in [item.name for item in POLICIES]:
        group = trades[trades["policy"] == policy].copy()
        cal = calibration[(calibration["policy"] == policy) & (calibration["is_oos"])].copy()
        executable = cal[cal["skip_reason"] == ""]
        if group.empty:
            rows.append({
                "policy": policy,
                "candidate_oos_count": int(len(cal)),
                "trade_count": 0,
                "trade_rate": 0.0,
                "win_rate": np.nan,
                "total_stake": 0.0,
                "total_pnl": 0.0,
                "roi_on_stake": np.nan,
                "final_equity": INITIAL_BANKROLL,
                "max_drawdown": np.nan,
                "profit_factor": np.nan,
                "avg_p_hat": np.nan,
                "avg_edge": np.nan,
                "avg_stake": np.nan,
                "longest_loss_streak": 0,
                "edge_pass_count": int(len(executable)),
            })
            continue
        ordered = group.sort_values(["settlement_time", "signal_id"])
        wins = ordered["won"].astype(bool)
        equity = INITIAL_BANKROLL + ordered["pnl"].cumsum()
        gross_profit = ordered.loc[ordered["pnl"] > 0, "pnl"].sum()
        gross_loss = -ordered.loc[ordered["pnl"] < 0, "pnl"].sum()
        rows.append({
            "policy": policy,
            "candidate_oos_count": int(len(cal)),
            "trade_count": int(len(ordered)),
            "trade_rate": float(len(ordered) / len(cal)) if len(cal) else np.nan,
            "win_rate": float(wins.mean()),
            "total_stake": float(ordered["stake"].sum()),
            "total_pnl": float(ordered["pnl"].sum()),
            "roi_on_stake": float(ordered["pnl"].sum() / ordered["stake"].sum()) if ordered["stake"].sum() else np.nan,
            "final_equity": float(equity.iloc[-1]),
            "max_drawdown": max_drawdown(equity),
            "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
            "avg_p_hat": float(ordered["p_hat"].mean()),
            "avg_edge": float(ordered["edge"].mean()),
            "avg_stake": float(ordered["stake"].mean()),
            "longest_loss_streak": longest_streak(wins, False),
            "edge_pass_count": int(len(executable)),
        })
    return pd.DataFrame(rows)


def plot_equity(equity: pd.DataFrame, output_path: Path) -> None:
    if equity.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for policy, group in equity.groupby("policy", dropna=False):
        ordered = group.sort_values("settlement_time")
        x = pd.to_datetime(ordered["settlement_time"])
        axes[0].plot(x, ordered["equity"], label=policy)
        axes[1].plot(x, ordered["drawdown"] * 100.0, label=policy)
    axes[0].set_title("Live-Style OOS Equity Curves")
    axes[0].set_ylabel("Bankroll")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Settlement Time")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def pct(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value * 100.0:.2f}%"


def report_text(summary: pd.DataFrame, validation: dict) -> str:
    lines = []
    lines.append("# Polymarket Live-Style Backtest v1")
    lines.append("")
    lines.append("## 结论先说")
    lines.append("")
    lines.append("- 现有 v17 信号有统计优势，但直接全信号无脑买不够实盘；优势主要来自特定方向、周期和入场类型。")
    lines.append("- 真实 Polymarket 的关键不是胜率本身，而是 `预测胜率 - 实际买入成本` 是否足够大。")
    lines.append("- 本脚本只用入场前已经结算的历史信号估计胜率，并从 2025-07-01 后开始 OOS 模拟，避免用未来结果选单笔交易。")
    lines.append("- 因为没有真实历史盘口，本回测仍然是保守合成成交价模型，不等于最终实盘结论。")
    lines.append("")
    lines.append("## 数据校验")
    lines.append("")
    lines.append(f"- 15m rows: {validation['rows']}")
    lines.append(f"- Data range: {validation['start']} to {validation['end']}")
    lines.append(f"- Bad 15m spacing count: {validation['bad_spacing_count']}")
    lines.append(f"- OOS start: {TRAIN_END}")
    lines.append("")
    lines.append("## Live-Style Policy Results")
    lines.append("")
    lines.append("| Policy | OOS candidates | Trades | Trade rate | Win rate | Avg p_hat | Avg edge | ROI stake | PnL | Final equity | Max DD | PF | Avg stake |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary.sort_values("total_pnl", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.policy} | {int(row.candidate_oos_count)} | {int(row.trade_count)} | {pct(row.trade_rate)} | "
            f"{pct(row.win_rate)} | {pct(row.avg_p_hat)} | {pct(row.avg_edge)} | {pct(row.roi_on_stake)} | "
            f"{row.total_pnl:.2f} | {row.final_equity:.2f} | {pct(row.max_drawdown)} | {row.profit_factor:.3f} | {row.avg_stake:.2f} |"
        )
    lines.append("")
    lines.append("## 实盘判断")
    lines.append("")
    lines.append("- 如果真实买入价经常在 `0.51-0.52`，并且只做高 edge 的 60m/360m/720m 子集，策略有小资金试运行价值。")
    lines.append("- 如果真实买入价常在 `0.55+`，大部分 v17 原始信号优势会被吃掉，不建议实盘无过滤追单。")
    lines.append("- 赚钱更多的方向：只交易滚动历史胜率足够高、样本数足够、edge 足够的信号；优先 long 360m 和 deep 720m 这类高质量子集。")
    lines.append("- 亏更少的方向：单周期最多 1 笔、同向锁仓、单笔 0.6%-1.0% bankroll、小 Kelly、回撤达到 10%-15% 暂停。")
    lines.append("- 下一步最好接入真实 Polymarket 历史盘口/成交价，否则所有价格模型都只能算压力测试。")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `live_style_trades_v1.csv`")
    lines.append("- `live_style_policy_calibration_v1.csv`")
    lines.append("- `live_style_summary_v1.csv`")
    lines.append("- `live_style_equity_curves_v1.csv`")
    lines.append("- `live_style_equity_comparison_v1.png`")
    lines.append("- `live_style_summary_v1.json`")
    return "\n".join(lines)


def main() -> None:
    LIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_15m_data()
    validation = validate_15m_data(prices)
    raw_bets = ensure_raw_binary_bets()
    all_trades = []
    all_calibration = []
    for policy in POLICIES:
        trades, calibration = simulate_policy(raw_bets, policy)
        all_trades.append(trades)
        all_calibration.append(calibration)
    trades_df = pd.concat(all_trades, ignore_index=True, sort=False) if all_trades else pd.DataFrame()
    calibration_df = pd.concat(all_calibration, ignore_index=True, sort=False) if all_calibration else pd.DataFrame()
    summary = summarize_live_trades(trades_df, calibration_df)
    equity = build_equity(trades_df)

    trades_df.to_csv(TRADES_FILE, index=False)
    calibration_df.to_csv(CALIBRATION_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    equity.to_csv(EQUITY_FILE, index=False)
    plot_equity(equity, PLOT_FILE)
    JSON_FILE.write_text(json.dumps({
        "validation": validation,
        "train_end": TRAIN_END,
        "policies": [policy.__dict__ for policy in POLICIES],
        "summary": json.loads(summary.to_json(orient="records")),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    report = report_text(summary, validation)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
