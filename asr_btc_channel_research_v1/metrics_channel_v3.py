from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def compute_max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def compute_summary(equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float) -> dict:
    equity = equity_df["equity"].astype(float)
    returns = equity.pct_change().fillna(0.0)
    total_return = equity.iloc[-1] / initial_capital - 1.0 if not equity.empty else 0.0
    bars_per_year = 365.0 * 24.0 * 4.0
    ann_return = (1.0 + total_return) ** (bars_per_year / max(len(equity), 1)) - 1.0 if len(equity) > 1 else 0.0
    ann_vol = float(returns.std(ddof=0) * np.sqrt(bars_per_year)) if len(returns) > 1 else 0.0
    sharpe = float((returns.mean() / returns.std(ddof=0)) * np.sqrt(bars_per_year)) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0
    max_dd = compute_max_drawdown(equity)
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0

    wins = trades_df[trades_df["pnl_abs"] > 0]
    losses = trades_df[trades_df["pnl_abs"] < 0]
    gross_profit = float(wins["pnl_abs"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl_abs"].sum()) if not losses.empty else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "volatility": float(ann_vol),
        "trade_count": int(len(trades_df)),
        "win_rate": float(len(wins) / len(trades_df)) if len(trades_df) else 0.0,
        "profit_factor": float(profit_factor),
        "avg_trade_pnl": float(trades_df["pnl_abs"].mean()) if len(trades_df) else 0.0,
        "avg_trade_return_pct": float(trades_df["pnl_pct"].mean()) if len(trades_df) else 0.0,
        "avg_win": float(wins["pnl_abs"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["pnl_abs"].mean()) if len(losses) else 0.0,
        "long_trade_count": int((trades_df["side"] == "long").sum()) if len(trades_df) else 0,
        "short_trade_count": int((trades_df["side"] == "short").sum()) if len(trades_df) else 0,
    }


def save_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
