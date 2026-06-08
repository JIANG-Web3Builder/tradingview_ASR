from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    ax.set_title("ASR BTC Equity Curve Comparison")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)

