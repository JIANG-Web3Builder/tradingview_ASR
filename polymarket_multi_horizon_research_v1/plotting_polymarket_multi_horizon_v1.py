from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from market_simulator_polymarket_multi_horizon_v1 import build_equity_curve


def _filter_variant(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "all_signals":
        return df.copy()
    if variant == "long_only":
        return df[df["side"] == "long"].copy()
    if variant == "short_only":
        return df[df["side"] == "short"].copy()
    if variant == "deep_entries":
        return df[df["entry_id"].isin(["Long2", "Long3", "Short2", "Short3"])].copy()
    return df.iloc[0:0].copy()


def plot_horizon_comparison(horizon_summary: pd.DataFrame, output_path: Path) -> None:
    if horizon_summary.empty:
        return
    subset = horizon_summary[
        (horizon_summary["sample_split"] == "ALL")
        & (horizon_summary["concurrency_mode"] == "independent")
        & (horizon_summary["buy_price"] == 0.5)
        & (horizon_summary["fee_drag"] == 0.0)
    ].copy()
    if subset.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for variant in ["all_signals", "long_only", "short_only", "deep_entries"]:
        data = subset[subset["variant"] == variant].sort_values("horizon_minutes")
        if data.empty:
            continue
        axes[0].plot(data["horizon_minutes"], data["win_rate"] * 100.0, marker="o", label=variant)
        axes[1].plot(data["horizon_minutes"], data["roi_on_stake"] * 100.0, marker="o", label=variant)
    axes[0].axhline(50.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_title("Strategy Accuracy by Holding Horizon | Price 0.50, Fee 0, Independent Signals")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Horizon Minutes")
    axes[1].set_ylabel("ROI on Stake (%)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_combined_equity_curves(priced_bets: pd.DataFrame, output_path: Path) -> None:
    if priced_bets.empty:
        return
    variants = ["all_signals", "long_only", "short_only", "deep_entries"]
    horizons = [60, 360, 1440]
    fig, axes = plt.subplots(len(horizons), 1, figsize=(13, 10), sharex=True)
    for ax, horizon in zip(axes, horizons):
        for variant in variants:
            data = _filter_variant(priced_bets, variant)
            data = data[
                (data["horizon_minutes"] == horizon)
                & (data["concurrency_mode"] == "independent")
                & (data["buy_price"] == 0.5)
                & (data["fee_drag"] == 0.0)
            ].copy()
            if data.empty:
                continue
            curve = build_equity_curve(data)
            if curve.empty:
                continue
            ax.plot(pd.to_datetime(curve["settlement_time"]), curve["equity"], label=variant)
        ax.set_title(f"{horizon}m Horizon")
        ax.set_ylabel("Bankroll")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Settlement Time")
    fig.suptitle("Combined Equity Curves | Price 0.50, Fee 0, Independent Signals", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_concurrency_comparison(price_sensitivity: pd.DataFrame, output_path: Path) -> None:
    if price_sensitivity.empty:
        return
    subset = price_sensitivity[
        (price_sensitivity["variant"] == "all_signals")
        & (price_sensitivity["sample_split"] == "ALL")
        & (price_sensitivity["horizon_minutes"].isin([60, 360, 1440]))
        & (price_sensitivity["buy_price"] == 0.5)
        & (price_sensitivity["fee_drag"] == 0.0)
    ].copy()
    if subset.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pivot_win = subset.pivot(index="horizon_minutes", columns="concurrency_mode", values="win_rate") * 100.0
    pivot_roi = subset.pivot(index="horizon_minutes", columns="concurrency_mode", values="roi_on_stake") * 100.0
    pivot_win.plot(kind="bar", ax=axes[0])
    pivot_roi.plot(kind="bar", ax=axes[1])
    axes[0].set_title("Win Rate by Concurrency Mode")
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].axhline(50.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_title("ROI by Concurrency Mode")
    axes[1].set_ylabel("ROI on Stake (%)")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xlabel("Horizon Minutes")
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_equity_curve(equity_curve: pd.DataFrame, output_path: Path, title: str) -> None:
    if equity_curve.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = pd.to_datetime(equity_curve["settlement_time"])
    axes[0].plot(x, equity_curve["equity"], label="Equity")
    axes[0].set_title(title)
    axes[0].set_ylabel("Bankroll")
    axes[0].grid(True, alpha=0.3)
    axes[1].fill_between(x, equity_curve["drawdown"] * 100.0, 0, color="red", alpha=0.3)
    axes[1].set_ylabel("DD %")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
