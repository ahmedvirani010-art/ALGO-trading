"""
Simple event-driven backtester for the RSI Divergence strategy.

Trade rules
-----------
Entry (long):  any bull signal fires (micro or macro, regular or hidden)
Entry (short): any bear signal fires (micro or macro, regular or hidden)
Exit:          opposite signal fires  — OR —  stop-loss / take-profit hit
Position:      one position at a time; new opposite signal closes current position
               then opens the new one in the same bar.

Default risk params
-------------------
stop_loss_pct  : 2 %   (e.g. buy at 100 → stop at 98)
take_profit_pct: 4 %   (e.g. buy at 100 → target at 104)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    direction: str          # "long" or "short"
    entry_date: object
    entry_price: float
    exit_date: Optional[object] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""   # "signal", "stop", "tp", "end"
    pnl_pct: float = 0.0

    def close(self, exit_date, exit_price: float, reason: str) -> None:
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        if self.direction == "long":
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price * 100
        else:
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price * 100


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)

    # ---- summary stats ----
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    long_trades: int = 0
    short_trades: int = 0

    def compute_stats(self) -> None:
        closed = [t for t in self.trades if t.exit_date is not None]
        self.total_trades = len(closed)
        if self.total_trades == 0:
            return

        pnls = [t.pnl_pct for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = self.winning_trades / self.total_trades * 100
        self.avg_win_pct = float(np.mean(wins)) if wins else 0.0
        self.avg_loss_pct = float(np.mean(losses)) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        self.total_return_pct = sum(pnls)
        self.long_trades = sum(1 for t in closed if t.direction == "long")
        self.short_trades = sum(1 for t in closed if t.direction == "short")

        # Max drawdown from equity curve
        if not self.equity_curve.empty:
            peak = self.equity_curve.cummax()
            dd = (self.equity_curve - peak) / peak * 100
            self.max_drawdown_pct = float(dd.min())

        # Annualised Sharpe (daily returns, rf=0)
        if len(pnls) > 1:
            arr = np.array(pnls)
            self.sharpe_ratio = float(arr.mean() / arr.std(ddof=1) * np.sqrt(252))

    def summary(self) -> str:
        lines = [
            "=" * 52,
            "         RSI DIVERGENCE STRATEGY — BACKTEST",
            "              National Foods (NATF)",
            "=" * 52,
            f"  Total trades     : {self.total_trades}",
            f"  Long / Short     : {self.long_trades} / {self.short_trades}",
            f"  Win rate         : {self.win_rate:.1f}%",
            f"  Winning trades   : {self.winning_trades}",
            f"  Losing trades    : {self.losing_trades}",
            f"  Avg win          : +{self.avg_win_pct:.2f}%",
            f"  Avg loss         : {self.avg_loss_pct:.2f}%",
            f"  Profit factor    : {self.profit_factor:.2f}",
            f"  Total return     : {self.total_return_pct:+.2f}%",
            f"  Max drawdown     : {self.max_drawdown_pct:.2f}%",
            f"  Sharpe ratio     : {self.sharpe_ratio:.2f}",
            "=" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

BULL_SIGNALS = [
    "micro_bull_regular", "micro_bull_hidden",
    "macro_bull_regular", "macro_bull_hidden",
]
BEAR_SIGNALS = [
    "micro_bear_regular", "micro_bear_hidden",
    "macro_bear_regular", "macro_bear_hidden",
]


def run(
    df: pd.DataFrame,
    stop_loss_pct: float = 2.0,
    take_profit_pct: float = 4.0,
) -> BacktestResult:
    """
    Run the backtest on *df* which must already contain the 8 signal columns
    produced by indicators.rsi_divergence.compute().

    Parameters
    ----------
    df              : DataFrame with OHLC + signal columns, sorted oldest→newest
    stop_loss_pct   : % below entry to place stop (long) / above entry (short)
    take_profit_pct : % above entry to place take-profit (long) / below (short)
    """
    result = BacktestResult()
    position: Optional[Trade] = None
    equity = 100.0          # start at 100 (percentage-based)
    equity_series: List[float] = []
    date_series: List = []

    for i, row in df.iterrows():
        date = row.name if hasattr(row, "name") else i
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        bull = any(bool(row.get(s, False)) for s in BULL_SIGNALS)
        bear = any(bool(row.get(s, False)) for s in BEAR_SIGNALS)

        if position is not None:
            # Check stop / TP using intrabar high/low
            if position.direction == "long":
                stop_price = position.entry_price * (1 - stop_loss_pct / 100)
                tp_price = position.entry_price * (1 + take_profit_pct / 100)
                if low <= stop_price:
                    exit_p = stop_price
                    position.close(date, exit_p, "stop")
                    equity += position.pnl_pct
                    result.trades.append(position)
                    position = None
                elif high >= tp_price:
                    exit_p = tp_price
                    position.close(date, exit_p, "tp")
                    equity += position.pnl_pct
                    result.trades.append(position)
                    position = None
                elif bear:
                    position.close(date, close, "signal")
                    equity += position.pnl_pct
                    result.trades.append(position)
                    position = None
            else:  # short
                stop_price = position.entry_price * (1 + stop_loss_pct / 100)
                tp_price = position.entry_price * (1 - take_profit_pct / 100)
                if high >= stop_price:
                    exit_p = stop_price
                    position.close(date, exit_p, "stop")
                    equity += position.pnl_pct
                    result.trades.append(position)
                    position = None
                elif low <= tp_price:
                    exit_p = tp_price
                    position.close(date, exit_p, "tp")
                    equity += position.pnl_pct
                    result.trades.append(position)
                    position = None
                elif bull:
                    position.close(date, close, "signal")
                    equity += position.pnl_pct
                    result.trades.append(position)
                    position = None

        # Open new position if flat and signal fires
        if position is None:
            if bull and not bear:
                position = Trade(direction="long", entry_date=date, entry_price=close)
            elif bear and not bull:
                position = Trade(direction="short", entry_date=date, entry_price=close)

        equity_series.append(equity)
        date_series.append(date)

    # Close any open position at last bar
    if position is not None:
        last_row = df.iloc[-1]
        position.close(df.index[-1], float(last_row["close"]), "end")
        equity += position.pnl_pct
        result.trades.append(position)
        equity_series[-1] = equity

    result.equity_curve = pd.Series(equity_series, index=date_series)
    result.compute_stats()
    return result
