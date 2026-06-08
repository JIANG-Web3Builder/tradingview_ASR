from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config_channel_v5 import INITIAL_CAPITAL

BASE_DIR = Path(r"D:\workspace\20260325\asr_btc_channel_research_v1")
OUTPUT_DIR = BASE_DIR / "output_channel_v5" / "position_sizing_leverage_v1"
BARS_PER_YEAR = 365.0 * 24.0 * 4.0
OOS_START = pd.Timestamp("2025-07-01 00:00:00")
MAINTENANCE_MARGIN_RATE = 0.005
MAX_REASONABLE_DD = -0.25
MIN_LIQUIDATION_BUFFER_PCT = 0.12


@dataclass(frozen=True)
class CandidateSource:
    label: str
    output_dir: Path
    equity_file: str
    trades_file: str
    summary_file: str


@dataclass(frozen=True)
class SizingPolicy:
    name: str
    mode: str
    base_notional_pct: float
    max_notional_pct: float
    dd_cut_1: float | None = None
    dd_mult_1: float = 1.0
    dd_cut_2: float | None = None
    dd_mult_2: float = 1.0
    target_ann_vol: float | None = None
    vol_lookback_bars: int = 96 * 30


CANDIDATES = [
    CandidateSource("v17", BASE_DIR / "output_channel_v4" / "v17", "equity_curve_channel_v4.csv", "trades_channel_v4.csv", "summary_channel_v4.json"),
    CandidateSource("v17_trend", BASE_DIR / "output_channel_v5" / "v17_trend", "equity_curve_channel_v5.csv", "trades_channel_v5.csv", "summary_channel_v5.json"),
    CandidateSource("v22", BASE_DIR / "output_channel_v5" / "v22", "equity_curve_channel_v5.csv", "trades_channel_v5.csv", "summary_channel_v5.json"),
    CandidateSource("v23", BASE_DIR / "output_channel_v5" / "v23", "equity_curve_channel_v5.csv", "trades_channel_v5.csv", "summary_channel_v5.json"),
    CandidateSource("v26", BASE_DIR / "output_channel_v5" / "v26", "equity_curve_channel_v5.csv", "trades_channel_v5.csv", "summary_channel_v5.json"),
    CandidateSource("v28", BASE_DIR / "output_channel_v5" / "v28", "equity_curve_channel_v5.csv", "trades_channel_v5.csv", "summary_channel_v5.json"),
    CandidateSource("v30", BASE_DIR / "output_channel_v5" / "v30", "equity_curve_channel_v5.csv", "trades_channel_v5.csv", "summary_channel_v5.json"),
]

FIXED_NOTIONAL_PCTS = [0.20, 0.30, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]
EXCHANGE_LEVERAGES = [1.0, 2.0, 3.0, 5.0, 10.0]
POLICIES = (
    [SizingPolicy(f"fixed_{pct:.2f}x", "fixed", pct, pct) for pct in FIXED_NOTIONAL_PCTS]
    + [
        SizingPolicy("dd_control_0.75x", "drawdown_control", 0.75, 0.75, -0.08, 0.65, -0.15, 0.35),
        SizingPolicy("dd_control_1.00x", "drawdown_control", 1.00, 1.00, -0.08, 0.65, -0.15, 0.35),
        SizingPolicy("dd_control_1.25x", "drawdown_control", 1.25, 1.25, -0.08, 0.60, -0.15, 0.30),
        SizingPolicy("vol_target_12pct", "vol_target", 1.00, 1.25, target_ann_vol=0.12),
        SizingPolicy("vol_target_16pct", "vol_target", 1.00, 1.50, target_ann_vol=0.16),
        SizingPolicy("vol_target_20pct", "vol_target", 1.00, 1.75, target_ann_vol=0.20),
    ]
)


def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())


def max_drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def summarize_equity(equity_df: pd.DataFrame, initial_capital: float) -> dict:
    equity = equity_df["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / initial_capital - 1.0) if len(equity) else 0.0
    ann_return = float((1.0 + total_return) ** (BARS_PER_YEAR / max(len(equity), 1)) - 1.0) if total_return > -1.0 and len(equity) > 1 else -1.0
    ann_vol = float(returns.std(ddof=0) * np.sqrt(BARS_PER_YEAR)) if len(returns) > 1 else 0.0
    sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(BARS_PER_YEAR)) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0
    max_dd = compute_max_drawdown(equity)
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "final_equity": float(equity.iloc[-1]) if len(equity) else initial_capital,
    }


def summarize_equity_array(equity: np.ndarray, initial_capital: float) -> dict:
    if len(equity) == 0:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "final_equity": initial_capital,
        }
    returns = np.zeros(len(equity), dtype=float)
    returns[1:] = equity[1:] / equity[:-1] - 1.0
    total_return = float(equity[-1] / initial_capital - 1.0)
    ann_return = float((1.0 + total_return) ** (BARS_PER_YEAR / max(len(equity), 1)) - 1.0) if total_return > -1.0 and len(equity) > 1 else -1.0
    ret_std = float(np.std(returns)) if len(returns) > 1 else 0.0
    ann_vol = float(ret_std * np.sqrt(BARS_PER_YEAR)) if len(returns) > 1 else 0.0
    sharpe = float((np.mean(returns) / ret_std) * np.sqrt(BARS_PER_YEAR)) if ret_std > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(np.min(dd))
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "final_equity": float(equity[-1]),
    }


def load_source(source: CandidateSource) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    equity_df = pd.read_csv(source.output_dir / source.equity_file, parse_dates=["open_time"])
    trades_df = pd.read_csv(source.output_dir / source.trades_file, parse_dates=["entry_time", "exit_time"])
    summary = json.loads((source.output_dir / source.summary_file).read_text(encoding="utf-8"))
    return equity_df, trades_df, summary


def estimate_exposure(equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.Series:
    exposure = pd.Series(0.0, index=equity_df.index)
    if trades_df.empty:
        return exposure
    times = equity_df["open_time"].to_numpy()
    for trade in trades_df.itertuples(index=False):
        entry_idx = int(np.searchsorted(times, np.datetime64(trade.entry_time), side="left"))
        exit_idx = int(np.searchsorted(times, np.datetime64(trade.exit_time), side="right"))
        qty = float(trade.qty)
        entry_price = float(trade.entry_price)
        notional = abs(qty * entry_price)
        base_equity = float(equity_df.loc[entry_idx, "equity"]) if entry_idx < len(equity_df) else INITIAL_CAPITAL
        exposure_pct = notional / base_equity if base_equity > 0 else 0.0
        if exit_idx > entry_idx:
            exposure.iloc[entry_idx:exit_idx] += exposure_pct
    return exposure.clip(lower=0.0)


def policy_multiplier(policy: SizingPolicy, base_returns: pd.Series, scaled_equity: list[float]) -> float:
    if policy.mode == "fixed":
        return policy.base_notional_pct
    current_equity = scaled_equity[-1]
    peak_equity = max(scaled_equity)
    dd = current_equity / peak_equity - 1.0 if peak_equity > 0 else 0.0
    if policy.mode == "drawdown_control":
        multiplier = policy.base_notional_pct
        if policy.dd_cut_2 is not None and dd <= policy.dd_cut_2:
            multiplier *= policy.dd_mult_2
        elif policy.dd_cut_1 is not None and dd <= policy.dd_cut_1:
            multiplier *= policy.dd_mult_1
        return min(multiplier, policy.max_notional_pct)
    if policy.mode == "vol_target":
        lookback = base_returns.iloc[max(0, len(scaled_equity) - policy.vol_lookback_bars) : len(scaled_equity)]
        realized_vol = float(lookback.std(ddof=0) * np.sqrt(BARS_PER_YEAR)) if len(lookback) > 10 else 0.0
        if realized_vol <= 0 or policy.target_ann_vol is None:
            return min(policy.base_notional_pct, policy.max_notional_pct)
        return min(policy.max_notional_pct, max(0.10, policy.target_ann_vol / realized_vol))
    raise ValueError(f"Unsupported policy mode: {policy.mode}")


def simulate_policy_once(equity_df: pd.DataFrame, policy: SizingPolicy) -> tuple[pd.DataFrame, dict, np.ndarray]:
    base_equity = equity_df["equity"].astype(float).to_numpy()
    base_returns = np.zeros(len(base_equity), dtype=float)
    base_returns[1:] = base_equity[1:] / base_equity[:-1] - 1.0

    if policy.mode == "fixed":
        multipliers = np.full(len(base_returns), policy.base_notional_pct, dtype=float)
        multipliers[0] = 0.0
        scaled_returns = base_returns * multipliers
        scaled_equity = INITIAL_CAPITAL * np.cumprod(1.0 + scaled_returns)
    elif policy.mode == "vol_target":
        base_return_series = pd.Series(base_returns)
        rolling_vol = base_return_series.rolling(policy.vol_lookback_bars, min_periods=11).std(ddof=0).to_numpy() * np.sqrt(BARS_PER_YEAR)
        multipliers = np.full(len(base_returns), policy.base_notional_pct, dtype=float)
        valid_vol = rolling_vol > 0
        multipliers[valid_vol] = policy.target_ann_vol / rolling_vol[valid_vol]
        multipliers = np.clip(multipliers, 0.10, policy.max_notional_pct)
        multipliers[0] = 0.0
        scaled_returns = base_returns * multipliers
        scaled_equity = INITIAL_CAPITAL * np.cumprod(1.0 + scaled_returns)
    elif policy.mode == "drawdown_control":
        scaled_equity = np.empty(len(base_returns), dtype=float)
        multipliers = np.empty(len(base_returns), dtype=float)
        scaled_equity[0] = INITIAL_CAPITAL
        multipliers[0] = 0.0
        peak_equity = INITIAL_CAPITAL
        for i in range(1, len(base_returns)):
            current_equity = scaled_equity[i - 1]
            dd = current_equity / peak_equity - 1.0 if peak_equity > 0 else 0.0
            mult = policy.base_notional_pct
            if policy.dd_cut_2 is not None and dd <= policy.dd_cut_2:
                mult *= policy.dd_mult_2
            elif policy.dd_cut_1 is not None and dd <= policy.dd_cut_1:
                mult *= policy.dd_mult_1
            mult = min(mult, policy.max_notional_pct)
            multipliers[i] = mult
            scaled_equity[i] = current_equity * (1.0 + base_returns[i] * mult)
            peak_equity = max(peak_equity, scaled_equity[i])
    else:
        raise ValueError(f"Unsupported policy mode: {policy.mode}")

    liquidated = bool(np.any(scaled_equity <= INITIAL_CAPITAL * 0.01))
    scaled_equity = np.maximum(scaled_equity, INITIAL_CAPITAL * 0.01)
    scaled_df = pd.DataFrame(
        {
            "open_time": equity_df["open_time"].reset_index(drop=True),
            "equity": scaled_equity,
            "notional_multiplier": multipliers,
        }
    )
    stats = summarize_equity_array(scaled_equity, INITIAL_CAPITAL)
    dd = scaled_equity / np.maximum.accumulate(scaled_equity) - 1.0
    stats.update(
        {
            "policy": policy.name,
            "policy_mode": policy.mode,
            "base_notional_pct": policy.base_notional_pct,
            "max_notional_pct": policy.max_notional_pct,
            "liquidated": liquidated,
            "days_below_10pct_dd": int(np.sum(dd <= -0.10) / 96),
            "days_below_20pct_dd": int(np.sum(dd <= -0.20) / 96),
        }
    )
    return scaled_df, stats, multipliers


def add_leverage_metrics(stats: dict, exposure: pd.Series, multipliers: np.ndarray, exchange_leverage: float) -> dict:
    exposure_values = exposure.to_numpy(dtype=float)
    effective_notional_pct = exposure_values * multipliers
    required_margin_pct = effective_notional_pct / exchange_leverage if exchange_leverage > 0 else np.full_like(effective_notional_pct, np.inf)
    active = effective_notional_pct > 0
    liquidation_move_pct = max(0.0, (1.0 / exchange_leverage) - MAINTENANCE_MARGIN_RATE) if np.any(active) else None
    row = dict(stats)
    row.update(
        {
            "exchange_leverage": exchange_leverage,
            "max_required_margin_pct": float(np.max(required_margin_pct)) if len(required_margin_pct) else 0.0,
            "avg_required_margin_pct": float(np.mean(required_margin_pct)) if len(required_margin_pct) else 0.0,
            "min_one_bar_liquidation_move_pct": float(liquidation_move_pct) if liquidation_move_pct is not None else None,
        }
    )
    return row


def policy_multiplier_fast(policy: SizingPolicy, base_returns: pd.Series, bar_index: int, current_equity: float, peak_equity: float) -> float:
    if policy.mode == "fixed":
        return policy.base_notional_pct
    dd = current_equity / peak_equity - 1.0 if peak_equity > 0 else 0.0
    if policy.mode == "drawdown_control":
        multiplier = policy.base_notional_pct
        if policy.dd_cut_2 is not None and dd <= policy.dd_cut_2:
            multiplier *= policy.dd_mult_2
        elif policy.dd_cut_1 is not None and dd <= policy.dd_cut_1:
            multiplier *= policy.dd_mult_1
        return min(multiplier, policy.max_notional_pct)
    if policy.mode == "vol_target":
        lookback = base_returns.iloc[max(0, bar_index - policy.vol_lookback_bars) : bar_index]
        realized_vol = float(lookback.std(ddof=0) * np.sqrt(BARS_PER_YEAR)) if len(lookback) > 10 else 0.0
        if realized_vol <= 0 or policy.target_ann_vol is None:
            return min(policy.base_notional_pct, policy.max_notional_pct)
        return min(policy.max_notional_pct, max(0.10, policy.target_ann_vol / realized_vol))
    raise ValueError(f"Unsupported policy mode: {policy.mode}")


def simulate_policy(equity_df: pd.DataFrame, exposure: pd.Series, policy: SizingPolicy, exchange_leverage: float) -> tuple[pd.DataFrame, dict]:
    base_equity = equity_df["equity"].astype(float).reset_index(drop=True)
    base_returns = base_equity.pct_change().fillna(0.0)
    scaled_equity = [INITIAL_CAPITAL]
    multipliers = [0.0]
    required_margin_pcts = [0.0]
    liq_buffers = [np.inf]
    liquidated = False
    peak_equity = INITIAL_CAPITAL

    for i in range(1, len(base_returns)):
        prev_equity = scaled_equity[-1]
        mult = policy_multiplier_fast(policy, base_returns, i, prev_equity, peak_equity)
        next_equity = prev_equity * (1.0 + float(base_returns.iloc[i]) * mult)
        required_margin_pct = float(exposure.iloc[i]) * mult / exchange_leverage if exchange_leverage > 0 else np.inf
        effective_notional_pct = float(exposure.iloc[i]) * mult
        liquidation_move_pct = max(0.0, (1.0 / exchange_leverage) - MAINTENANCE_MARGIN_RATE) if effective_notional_pct > 0 else np.inf
        if next_equity <= INITIAL_CAPITAL * 0.01:
            next_equity = INITIAL_CAPITAL * 0.01
            liquidated = True
        peak_equity = max(peak_equity, next_equity)
        scaled_equity.append(next_equity)
        multipliers.append(mult)
        required_margin_pcts.append(required_margin_pct)
        liq_buffers.append(liquidation_move_pct)

    scaled_df = pd.DataFrame(
        {
            "open_time": equity_df["open_time"].reset_index(drop=True),
            "equity": scaled_equity,
            "notional_multiplier": multipliers,
            "estimated_base_exposure_pct": exposure.reset_index(drop=True),
            "required_margin_pct": required_margin_pcts,
            "one_bar_liquidation_move_pct": liq_buffers,
        }
    )
    stats = summarize_equity(scaled_df, INITIAL_CAPITAL)
    dd_series = max_drawdown_series(scaled_df["equity"])
    margin = scaled_df["required_margin_pct"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    liq = scaled_df["one_bar_liquidation_move_pct"].replace([np.inf, -np.inf], np.nan)
    stats.update(
        {
            "policy": policy.name,
            "policy_mode": policy.mode,
            "base_notional_pct": policy.base_notional_pct,
            "max_notional_pct": policy.max_notional_pct,
            "exchange_leverage": exchange_leverage,
            "max_required_margin_pct": float(margin.max()),
            "avg_required_margin_pct": float(margin.mean()),
            "min_one_bar_liquidation_move_pct": float(liq.min()) if liq.notna().any() else None,
            "liquidated": bool(liquidated),
            "days_below_10pct_dd": int((dd_series <= -0.10).sum() / 96),
            "days_below_20pct_dd": int((dd_series <= -0.20).sum() / 96),
        }
    )
    return scaled_df, stats


def score_row(row: dict) -> float:
    if row["liquidated"]:
        return -999.0
    if row["max_drawdown"] < -0.35:
        return -50.0 + row["sharpe"]
    dd_penalty = max(0.0, abs(row["max_drawdown"]) - 0.18) * 4.0
    margin_penalty = max(0.0, row["max_required_margin_pct"] - 0.65) * 1.5
    return float(row["sharpe"] + row["calmar"] * 0.15 + row["annualized_return"] * 0.5 - dd_penalty - margin_penalty)


def build_report(results_df: pd.DataFrame, source_summaries: list[dict]) -> str:
    viable = results_df[
        (results_df["liquidated"] == False)
        & (results_df["max_drawdown"] >= MAX_REASONABLE_DD)
        & (results_df["min_one_bar_liquidation_move_pct"].fillna(1.0) >= MIN_LIQUIDATION_BUFFER_PCT)
    ].copy()
    top_all = results_df.sort_values("score", ascending=False).head(15)
    top_viable = viable.sort_values("score", ascending=False).head(15)
    lines = [
        "# BTC Channel Position Sizing and Leverage Study v1",
        "",
        "## Method",
        "- Uses existing strategy equity/trade outputs without changing signal logic.",
        "- Treats current backtests as 1.0x notional reference curves, then rescales bar returns by target notional percentage.",
        "- Exchange leverage is evaluated as margin usage and liquidation buffer, not as an independent source of edge.",
        "- Assumes BTCUSDT 15m data from 2024-01-01 to 2026-05-01 and maintenance margin rate 0.5%.",
        "",
        "## Source Strategy Baselines",
        "| version | return | sharpe | max_dd | trades | win_rate | pf |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in source_summaries:
        lines.append(
            f"| {s['version']} | {s['total_return']:.2%} | {s['sharpe']:.3f} | {s['max_drawdown']:.2%} | {int(s['trade_count'])} | {s['win_rate']:.2%} | {s['profit_factor']:.3f} |"
        )
    lines.extend(["", "## Top Viable Policies", "| rank | version | policy | exchange_lev | return | ann_ret | sharpe | max_dd | margin_max | liq_buffer | score |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for rank, row in enumerate(top_viable.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.version} | {row.policy} | {row.exchange_leverage:.1f} | {row.total_return:.2%} | {row.annualized_return:.2%} | {row.sharpe:.3f} | {row.max_drawdown:.2%} | {row.max_required_margin_pct:.2%} | {row.min_one_bar_liquidation_move_pct:.2%} | {row.score:.3f} |"
        )
    lines.extend(["", "## Top Raw Scores", "| rank | version | policy | exchange_lev | return | sharpe | max_dd | margin_max | liq_buffer | score |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for rank, row in enumerate(top_all.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.version} | {row.policy} | {row.exchange_leverage:.1f} | {row.total_return:.2%} | {row.sharpe:.3f} | {row.max_drawdown:.2%} | {row.max_required_margin_pct:.2%} | {row.min_one_bar_liquidation_move_pct:.2%} | {row.score:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Practical Recommendation",
            "- Prefer strategy version v28 or v26 if continuing with the v17-like family; v30 is more conservative by drawdown but gives up return.",
            "- For live trading, target 0.75x to 1.00x total notional exposure first. Move toward 1.25x only after paper/live fills match backtest slippage for several months.",
            "- Set exchange leverage around 2x or 3x. This keeps margin usage efficient while leaving large liquidation distance; avoid using 5x-10x unless isolated margin and strict position caps are enforced.",
            "- Use drawdown control: cut notional to about 60%-65% of normal after an 8% equity drawdown, and to about 30%-35% after a 15% drawdown. Resume normal size only after a new equity high or a manual review.",
            "- Do not size from the best raw return alone. Policies above 1.25x begin to convert an otherwise strong strategy into a high psychological and operational risk profile.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    source_rows: list[dict] = []
    best_curves: dict[str, pd.DataFrame] = {}

    for source in CANDIDATES:
        equity_df, trades_df, source_summary = load_source(source)
        source_row = {
            "version": source.label,
            "total_return": float(source_summary["total_return"]),
            "sharpe": float(source_summary["sharpe"]),
            "max_drawdown": float(source_summary["max_drawdown"]),
            "trade_count": int(source_summary["trade_count"]),
            "win_rate": float(source_summary["win_rate"]),
            "profit_factor": float(source_summary["profit_factor"]),
        }
        source_rows.append(source_row)
        exposure = estimate_exposure(equity_df, trades_df)
        for policy in POLICIES:
            scaled_df, base_stats, multipliers = simulate_policy_once(equity_df, policy)
            for exchange_leverage in EXCHANGE_LEVERAGES:
                stats = add_leverage_metrics(base_stats, exposure, multipliers, exchange_leverage)
                stats["version"] = source.label
                stats["score"] = score_row(stats)
                result_rows.append(stats)
                key = f"{source.label}_{policy.name}_{exchange_leverage:g}x"
                if policy.name in {"fixed_0.75x", "fixed_1.00x", "dd_control_1.00x"} and exchange_leverage in {2.0, 3.0}:
                    best_curves[key] = scaled_df[["open_time", "equity"]].copy()

    results_df = pd.DataFrame(result_rows)
    source_df = pd.DataFrame(source_rows).sort_values(["sharpe", "total_return"], ascending=False)
    results_df = results_df.sort_values("score", ascending=False)
    source_df.to_csv(OUTPUT_DIR / "source_strategy_summary_v1.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "position_sizing_leverage_scan_v1.csv", index=False)

    viable_df = results_df[
        (results_df["liquidated"] == False)
        & (results_df["max_drawdown"] >= MAX_REASONABLE_DD)
        & (results_df["min_one_bar_liquidation_move_pct"].fillna(1.0) >= MIN_LIQUIDATION_BUFFER_PCT)
    ].copy()
    viable_df.to_csv(OUTPUT_DIR / "viable_position_sizing_leverage_v1.csv", index=False)

    curve_dir = OUTPUT_DIR / "selected_equity_curves"
    curve_dir.mkdir(exist_ok=True)
    for name, curve in best_curves.items():
        curve.to_csv(curve_dir / f"{name}.csv", index=False)

    report = build_report(results_df, source_rows)
    (OUTPUT_DIR / "position_sizing_leverage_report_v1.md").write_text(report, encoding="utf-8")
    metadata = {
        "candidate_count": len(CANDIDATES),
        "policy_count": len(POLICIES),
        "exchange_leverage_count": len(EXCHANGE_LEVERAGES),
        "scan_count": int(len(results_df)),
        "viable_count": int(len(viable_df)),
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "metadata_v1.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(results_df.head(10)[["version", "policy", "exchange_leverage", "total_return", "sharpe", "max_drawdown", "max_required_margin_pct", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
