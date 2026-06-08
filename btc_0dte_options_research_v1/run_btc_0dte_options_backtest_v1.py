from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config_btc_0dte_options_v1 import (
    BINANCE_ENTRY_SLIPPAGE_RATE,
    BINANCE_OPTION_FEE_CAP_RATE,
    BINANCE_EXIT_SLIPPAGE_RATE,
    BINANCE_OPTION_TAKER_FEE_RATE,
    DATA_SYNTHETIC_DIR,
    ENTRY_DELAY_MINUTES,
    FIXED_PREMIUM_STAKE_USD,
    HOLD_HOURS,
    INITIAL_BANKROLL,
    MAX_OPEN_POSITIONS,
    OUTPUT_DIR,
    SYNTHETIC_OPTION_QUOTES_FILE,
    SYNTHETIC_OPTION_TRADES_FILE,
    VARIANTS,
)
from data_loader_btc_0dte_options_v1 import load_btc_15m, load_v17_signals, validate_btc_15m
from synthetic_option_pricing_v1 import (
    add_realized_volatility,
    atm_strike,
    choose_option_type,
    conservative_option_quote,
    next_daily_expiry,
)


def nearest_price_row(price_data: pd.DataFrame, target_time: pd.Timestamp) -> pd.Series | None:
    eligible = price_data[price_data["open_time"] >= target_time]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def build_synthetic_option_candidates(signals: pd.DataFrame, price_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    quote_rows: list[dict] = []

    for signal in signals.itertuples(index=False):
        signal_time = pd.Timestamp(signal.entry_time)
        entry_target_time = signal_time + pd.Timedelta(minutes=ENTRY_DELAY_MINUTES)
        exit_target_time = entry_target_time + pd.Timedelta(hours=HOLD_HOURS)
        entry_row = nearest_price_row(price_data, entry_target_time)
        exit_row = nearest_price_row(price_data, exit_target_time)
        if entry_row is None or exit_row is None:
            continue

        entry_time = pd.Timestamp(entry_row["open_time"])
        exit_time = pd.Timestamp(exit_row["open_time"])
        entry_spot = float(entry_row["close"])
        exit_spot = float(exit_row["close"])
        expiry_time = next_daily_expiry(entry_time)
        if expiry_time is None or exit_time >= expiry_time:
            continue

        option_type = choose_option_type(signal.side)
        strike = atm_strike(entry_spot)
        entry_quote = conservative_option_quote(
            spot=entry_spot,
            strike=strike,
            expiry_time=expiry_time,
            quote_time=entry_time,
            rv_base=float(entry_row["rv_base"]),
            option_type=option_type,
            phase="entry",
        )
        exit_quote = conservative_option_quote(
            spot=exit_spot,
            strike=strike,
            expiry_time=expiry_time,
            quote_time=exit_time,
            rv_base=float(exit_row["rv_base"]),
            option_type=option_type,
            phase="exit",
        )

        entry_price_after_slippage = entry_quote["worst_executable_price"] * (1.0 + BINANCE_ENTRY_SLIPPAGE_RATE)
        exit_price_after_slippage = exit_quote["worst_executable_price"] * (1.0 - BINANCE_EXIT_SLIPPAGE_RATE)
        contracts = FIXED_PREMIUM_STAKE_USD / entry_price_after_slippage if entry_price_after_slippage > 0 else 0.0
        entry_premium_value = contracts * entry_price_after_slippage
        exit_premium_value = contracts * max(exit_price_after_slippage, 0.0)
        entry_underlying_notional = contracts * entry_spot
        exit_underlying_notional = contracts * exit_spot
        entry_fee = min(entry_underlying_notional * BINANCE_OPTION_TAKER_FEE_RATE, entry_premium_value * BINANCE_OPTION_FEE_CAP_RATE)
        exit_fee = min(exit_underlying_notional * BINANCE_OPTION_TAKER_FEE_RATE, exit_premium_value * BINANCE_OPTION_FEE_CAP_RATE)
        total_cost = entry_premium_value + entry_fee
        pnl = exit_premium_value - exit_fee - total_cost
        return_on_cost = pnl / total_cost if total_cost > 0 else np.nan
        direction_return_pct = (exit_spot / entry_spot - 1.0) * 100.0 * (1 if str(signal.side) == "long" else -1)

        common = {
            "signal_id": int(signal.signal_id),
            "trade_id": int(signal.trade_id),
            "side": str(signal.side),
            "entry_id": str(signal.entry_id),
            "signal_time": signal_time,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "expiry_time": expiry_time,
            "option_type": option_type,
            "strike": float(strike),
        }
        for q in [entry_quote, exit_quote]:
            quote_rows.append({**common, **q})

        rows.append(
            {
                **common,
                "entry_spot": entry_spot,
                "exit_spot": exit_spot,
                "direction_return_pct": direction_return_pct,
                "entry_tte_hours": float(entry_quote["tte_hours"]),
                "exit_tte_hours": float(exit_quote["tte_hours"]),
                "entry_rv_base": float(entry_quote["rv_base"]),
                "exit_rv_base": float(exit_quote["rv_base"]),
                "entry_iv_used": float(entry_quote["iv_used"]),
                "exit_iv_used": float(exit_quote["iv_used"]),
                "entry_theoretical_mid": float(entry_quote["theoretical_mid"]),
                "exit_theoretical_mid": float(exit_quote["theoretical_mid"]),
                "entry_intrinsic": float(entry_quote["intrinsic"]),
                "exit_intrinsic": float(exit_quote["intrinsic"]),
                "entry_bid": float(entry_quote["bid"]),
                "entry_ask": float(entry_quote["ask"]),
                "exit_bid": float(exit_quote["bid"]),
                "exit_ask": float(exit_quote["ask"]),
                "entry_worst_model_price": float(entry_quote["worst_executable_price"]),
                "exit_worst_model_price": float(exit_quote["worst_executable_price"]),
                "entry_price_after_binance_slippage": float(entry_price_after_slippage),
                "exit_price_after_binance_slippage": float(exit_price_after_slippage),
                "contracts": float(contracts),
                "entry_underlying_notional": float(entry_underlying_notional),
                "exit_underlying_notional": float(exit_underlying_notional),
                "entry_premium_value": float(entry_premium_value),
                "exit_premium_value": float(exit_premium_value),
                "entry_fee": float(entry_fee),
                "exit_fee": float(exit_fee),
                "total_cost": float(total_cost),
                "pnl": float(pnl),
                "return_on_cost": float(return_on_cost),
                "won": bool(pnl > 0),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(quote_rows)


def apply_variant(candidates: pd.DataFrame, variant: dict) -> pd.DataFrame:
    filtered = candidates.copy()
    sides = variant.get("sides")
    entry_ids = variant.get("entry_ids")
    if sides is not None:
        filtered = filtered[filtered["side"].isin(sides)].copy()
    if entry_ids is not None:
        filtered = filtered[filtered["entry_id"].isin(entry_ids)].copy()
    return filtered


def select_trades_with_concurrency(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    rows: list[pd.Series] = []
    open_positions: list[tuple[pd.Timestamp, float]] = []
    bankroll = INITIAL_BANKROLL
    for row in candidates.sort_values(["entry_time", "signal_id"]).itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time)
        still_open: list[tuple[pd.Timestamp, float]] = []
        for exit_time, pnl in open_positions:
            if exit_time <= entry_time:
                bankroll += pnl
            else:
                still_open.append((exit_time, pnl))
        open_positions = still_open
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue
        row_dict = row._asdict()
        if bankroll < float(row_dict["total_cost"]):
            continue
        rows.append(pd.Series(row_dict))
        open_positions.append((pd.Timestamp(row.exit_time), float(row.pnl)))
    return pd.DataFrame(rows)


def build_equity_curve(trades: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["variant", "exit_time", "pnl", "equity", "drawdown", "trade_count"])
    ordered = trades.sort_values(["exit_time", "signal_id"]).copy()
    curve = ordered.groupby("exit_time", as_index=False).agg(pnl=("pnl", "sum"), trade_count=("signal_id", "count"))
    curve["variant"] = variant_name
    curve["equity"] = INITIAL_BANKROLL + curve["pnl"].cumsum()
    curve["roll_max"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["roll_max"] - 1.0
    return curve[["variant", "exit_time", "pnl", "equity", "drawdown", "trade_count"]]


def max_drawdown(curve: pd.DataFrame) -> float:
    if curve.empty:
        return np.nan
    return float(curve["drawdown"].min())


def profit_factor(trades: pd.DataFrame) -> float:
    gains = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    losses = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def longest_loss_streak(trades: pd.DataFrame) -> int:
    longest = 0
    current = 0
    for won in trades.sort_values(["exit_time", "signal_id"])["won"].astype(bool):
        if won:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def summarize_variant(variant_name: str, candidates: pd.DataFrame, trades: pd.DataFrame, curve: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "variant": variant_name,
            "candidate_count": int(len(candidates)),
            "trade_count": 0,
            "trade_rate": 0.0,
            "win_rate": np.nan,
            "total_cost": 0.0,
            "total_pnl": 0.0,
            "roi_on_cost": np.nan,
            "final_equity": INITIAL_BANKROLL,
            "max_drawdown": np.nan,
            "profit_factor": np.nan,
            "avg_return_on_cost": np.nan,
            "median_return_on_cost": np.nan,
            "longest_loss_streak": 0,
            "avg_entry_iv": np.nan,
            "avg_exit_iv": np.nan,
        }
    total_cost = float(trades["total_cost"].sum())
    total_pnl = float(trades["pnl"].sum())
    return {
        "variant": variant_name,
        "candidate_count": int(len(candidates)),
        "trade_count": int(len(trades)),
        "trade_rate": float(len(trades) / len(candidates)) if len(candidates) else 0.0,
        "win_rate": float(trades["won"].mean()),
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "roi_on_cost": float(total_pnl / total_cost) if total_cost else np.nan,
        "final_equity": float(INITIAL_BANKROLL + total_pnl),
        "max_drawdown": max_drawdown(curve),
        "profit_factor": profit_factor(trades),
        "avg_return_on_cost": float(trades["return_on_cost"].mean()),
        "median_return_on_cost": float(trades["return_on_cost"].median()),
        "longest_loss_streak": longest_loss_streak(trades),
        "avg_entry_iv": float(trades["entry_iv_used"].mean()),
        "avg_exit_iv": float(trades["exit_iv_used"].mean()),
    }


def plot_equity_curves(curves: pd.DataFrame, output_path: Path) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for variant, group in curves.groupby("variant"):
        ax.plot(pd.to_datetime(group["exit_time"]), group["equity"], label=variant, linewidth=1.8)
    ax.axhline(INITIAL_BANKROLL, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title("BTC 0DTE Synthetic Worst-Case Option Backtest Equity")
    ax.set_xlabel("Exit Time")
    ax.set_ylabel("Equity USD")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(summary: pd.DataFrame, validation: dict, output_path: Path) -> None:
    ordered = summary.sort_values("total_pnl", ascending=False).copy()
    lines = [
        "# BTC 0DTE Synthetic Worst-Case Options Backtest v1",
        "",
        "## 口径",
        "",
        "- 使用 BTCUSDT 15m K 线合成 0DTE ATM 期权价格，不使用真实期权盘口。",
        "- 开仓买价按高 IV、高 ask、额外 markup、币安滑点和 taker fee 处理。",
        "- 平仓卖价按低 IV、低 bid、额外 haircut、币安滑点和 taker fee 处理。",
        "- v17 long 信号买 Call，short 信号买 Put，开仓延迟 15m，约 6h 平仓。",
        "- 每笔固定使用 100 USD 权利金，最多同时 1 笔持仓。",
        "",
        "## 数据校验",
        "",
        f"- BTC rows: {validation['rows']}",
        f"- BTC range: {validation['start']} to {validation['end']}",
        f"- Bad 15m spacing count: {validation['bad_15m_spacing_count']}",
        "",
        "## 结果汇总",
        "",
        "| Variant | Candidates | Trades | Win rate | ROI cost | PnL | Final equity | Max DD | PF | Avg entry IV | Avg exit IV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered.itertuples(index=False):
        win_rate = "nan" if pd.isna(row.win_rate) else f"{row.win_rate:.2%}"
        roi = "nan" if pd.isna(row.roi_on_cost) else f"{row.roi_on_cost:.2%}"
        dd = "nan" if pd.isna(row.max_drawdown) else f"{row.max_drawdown:.2%}"
        pf = "nan" if pd.isna(row.profit_factor) else f"{row.profit_factor:.3f}"
        avg_entry_iv = "nan" if pd.isna(row.avg_entry_iv) else f"{row.avg_entry_iv:.2%}"
        avg_exit_iv = "nan" if pd.isna(row.avg_exit_iv) else f"{row.avg_exit_iv:.2%}"
        lines.append(
            f"| {row.variant} | {row.candidate_count} | {row.trade_count} | {win_rate} | {roi} | {row.total_pnl:.2f} | {row.final_equity:.2f} | {dd} | {pf} | {avg_entry_iv} | {avg_exit_iv} |"
        )
    lines.extend(
        [
            "",
            "## 解读",
            "",
            "- 这是压力测试，不是真实期权盘口回放。",
            "- 如果在这种最恶劣模型下仍然盈利，策略才值得继续接真实期权数据验证。",
            "- 如果亏损，说明 0DTE 时间价值损耗、价差、IV crush、手续费和滑点会快速吞噬方向优势。",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    btc_raw = load_btc_15m()
    validation = validate_btc_15m(btc_raw)
    btc = add_realized_volatility(btc_raw)
    signals = load_v17_signals()

    candidates, quotes = build_synthetic_option_candidates(signals, btc)
    candidates.to_csv(SYNTHETIC_OPTION_TRADES_FILE, index=False)
    quotes.to_csv(SYNTHETIC_OPTION_QUOTES_FILE, index=False)

    all_trade_frames: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for variant_name, variant in VARIANTS.items():
        variant_candidates = apply_variant(candidates, variant)
        trades = select_trades_with_concurrency(variant_candidates)
        trades = trades.copy()
        trades["variant"] = variant_name
        curve = build_equity_curve(trades, variant_name)
        summaries.append(summarize_variant(variant_name, variant_candidates, trades, curve))
        all_trade_frames.append(trades)
        all_curves.append(curve)

    backtest_trades = pd.concat(all_trade_frames, ignore_index=True, sort=False) if all_trade_frames else pd.DataFrame()
    curves = pd.concat(all_curves, ignore_index=True, sort=False) if all_curves else pd.DataFrame()
    summary = pd.DataFrame(summaries)

    backtest_trades.to_csv(OUTPUT_DIR / "btc_0dte_option_backtest_trades_v1.csv", index=False)
    curves.to_csv(OUTPUT_DIR / "btc_0dte_option_equity_curves_v1.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "btc_0dte_option_summary_v1.csv", index=False)
    (OUTPUT_DIR / "btc_0dte_option_summary_v1.json").write_text(
        json.dumps(
            {
                "data_validation": validation,
                "synthetic_candidate_count": int(len(candidates)),
                "synthetic_quote_count": int(len(quotes)),
                "summary": summary.replace({np.nan: None, np.inf: "inf", -np.inf: "-inf"}).to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_equity_curves(curves, OUTPUT_DIR / "btc_0dte_option_equity_curves_v1.png")
    write_report(summary, validation, OUTPUT_DIR / "btc_0dte_option_research_report_v1.md")

    print(f"Saved synthetic option candidates: {SYNTHETIC_OPTION_TRADES_FILE}")
    print(f"Saved synthetic option quotes: {SYNTHETIC_OPTION_QUOTES_FILE}")
    print(f"Saved output directory: {OUTPUT_DIR}")
    print(summary.sort_values("total_pnl", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
