"""
Walk-forward portfolio simulation.

Rules
-----
- Capital: PKR 100,000 split equally into 3 slots (~33,333 each)
- Entry:   slot is empty AND the orchestrator ranks that stock in top 3
           AND an RSI divergence signal fires on that bar
- Exit:    (a) stop-loss 2% / take-profit 4%  (intrabar high/low)
           (b) within first 3 bars: stock drops out of orchestrator top-3
               → early exit at close (signal not holding)
           (c) within first 3 bars: opposite RSI divergence signal fires
               → exit and flip direction
- Replace: after any exit, scan current top-3 (excluding already-held tickers)
           for the next active signal bar
- Sizing:  full slot value per trade (compounding)

Usage:
    python portfolio_sim.py [--capital 100000] [--stop 2.0] [--tp 4.0]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config.rsi_div_config import Config
from indicators.rsi_divergence import compute, SIGNAL_COLUMNS
from indicators.ta_utils import (
    compute_rsi, compute_atr, compute_adx, compute_sma_distance,
)

# ---------------------------------------------------------------------------
# Best configs from grid-search optimisation
# ---------------------------------------------------------------------------

STOCKS: dict[str, str] = {
    "NATF":    "data/NATF_price_history_full_adj.csv",
    "BAFL":    "data/BAFL_price_history_adj.csv",
    "HUBC":    "data/HUBC_price_history_adj.csv",
    "ITTEHAD": "data/ITTEHAD_price_history_adj.csv",
    "LUCK":    "data/LUCK_price_history_adj.csv",
    "MARI":    "data/MARI_price_history_adj.csv",
}

BEST_CONFIGS: dict[str, dict] = {
    "NATF":    {"smooth_length":10,"p_len":1,"p_len_macro":3, "min_dist":5,"max_dist":100,"min_rsi_diff":4.0},
    "BAFL":    {"smooth_length":10,"p_len":1,"p_len_macro":10,"min_dist":5,"max_dist":100,"min_rsi_diff":4.0},
    "HUBC":    {"smooth_length":10,"p_len":1,"p_len_macro":10,"min_dist":5,"max_dist":100,"min_rsi_diff":4.0},
    "ITTEHAD": {"smooth_length":10,"p_len":1,"p_len_macro":5, "min_dist":5,"max_dist":100,"min_rsi_diff":1.0},
    "LUCK":    {"smooth_length":10,"p_len":2,"p_len_macro":5, "min_dist":3,"max_dist":100,"min_rsi_diff":1.0},
    "MARI":    {"smooth_length":10,"p_len":1,"p_len_macro":10,"min_dist":5,"max_dist":100,"min_rsi_diff":4.0},
}

# Historical Sharpe (from full grid search) used as fixed quality component
HIST_SHARPE: dict[str, float] = {
    "NATF": 4.25, "BAFL": 2.94, "HUBC": 5.12,
    "ITTEHAD": 3.37, "LUCK": 2.94, "MARI": 9.11,
}

BULL_COLS = [c for c in SIGNAL_COLUMNS if "bull" in c]
BEAR_COLS = [c for c in SIGNAL_COLUMNS if "bear" in c]

N_SLOTS = 3

# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True).set_index("date")
    for col in ["price", "open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    if "price" in df.columns and "close" not in df.columns:
        df = df.rename(columns={"price": "close"})
    return df[["open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Orchestrator scoring helpers
# ---------------------------------------------------------------------------

def _score_adx(v: float) -> int:
    if v > 40: return 20
    if v > 30: return 15
    if v > 25: return 10
    if v > 20: return 5
    return 0

def _score_dir(v: float) -> int:
    a = abs(v)
    if a > 20: return 20
    if a > 10: return 15
    if a > 5:  return 10
    if a > 2:  return 5
    return 0

def _score_rsi(v: float) -> int:
    d = abs(v - 50.0)
    if d >= 30: return 20   # RSI ≥80 or ≤20
    if d >= 20: return 10   # RSI ≥70 or ≤30
    if d >= 10: return 5
    return 0

def _score_atr(v: float) -> int:
    if v > 3.0: return 20
    if v > 2.0: return 15
    if v > 1.5: return 10
    if v > 1.0: return 5
    return 0

def _norm_sharpe(s: float, mn: float, mx: float) -> int:
    if mx == mn: return 10
    return int(round((s - mn) / (mx - mn) * 20))


# ---------------------------------------------------------------------------
# Precompute all series for every stock
# ---------------------------------------------------------------------------

def precompute(stop_pct: float, tp_pct: float) -> dict[str, dict]:
    """Return per-ticker dict with price df, signal df, and daily score series."""
    print("Pre-computing signals and indicator scores...")

    # Normalise historical Sharpe across all stocks
    sh_vals = list(HIST_SHARPE.values())
    sh_mn, sh_mx = min(sh_vals), max(sh_vals)

    stock_data: dict[str, dict] = {}

    for ticker, csv_path in STOCKS.items():
        print(f"  {ticker} ...", end="", flush=True)
        df = load_csv(csv_path)

        # Signals
        cfg = Config(**BEST_CONFIGS[ticker])
        df_sig = compute(df, cfg)

        # Live indicator scores (every bar)
        adx_df = compute_adx(df)
        rsi_s  = compute_rsi(df["close"], 14)
        atr_s  = compute_atr(df)
        dist_s = compute_sma_distance(df["close"], 200)
        atr_pct = atr_s / df["close"] * 100.0

        s_hist = _norm_sharpe(HIST_SHARPE[ticker], sh_mn, sh_mx)

        score_s = (
            adx_df["adx"].apply(_score_adx)
            + dist_s.apply(_score_dir)
            + rsi_s.apply(_score_rsi)
            + atr_pct.apply(_score_atr)
            + s_hist
        )

        stock_data[ticker] = {
            "df":    df,
            "sig":   df_sig[SIGNAL_COLUMNS],
            "score": score_s,
        }
        print(f" done")

    return stock_data


# ---------------------------------------------------------------------------
# Portfolio position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker:      str
    direction:   str        # "long" | "short"
    entry_date:  pd.Timestamp
    entry_bar:   int        # integer position in common date index
    entry_price: float
    stop_price:  float
    tp_price:    float


@dataclass
class ClosedTrade:
    ticker:     str
    direction:  str
    entry_date: pd.Timestamp
    exit_date:  pd.Timestamp
    entry_px:   float
    exit_px:    float
    pnl_pct:    float
    reason:     str
    slot_pnl:   float       # PKR P&L on that slot


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate(
    stock_data: dict[str, dict],
    initial_capital: float,
    stop_pct: float,
    tp_pct: float,
) -> tuple[list[ClosedTrade], pd.Series]:

    # Common trading dates = intersection across all stocks
    date_sets = [set(stock_data[t]["df"].index) for t in stock_data]
    common_dates = sorted(set.intersection(*date_sets))

    # Portfolio state
    slot_cash: list[float]             = [initial_capital / N_SLOTS] * N_SLOTS
    slot_pos:  list[Optional[Position]] = [None] * N_SLOTS

    equity_curve: list[float] = []
    closed_trades: list[ClosedTrade] = []

    for bar_idx, date in enumerate(common_dates):

        # ── orchestrator ranking for today ──────────────────────────────────
        today_scores: dict[str, float] = {}
        for ticker, d in stock_data.items():
            if date in d["score"].index and not np.isnan(d["score"].loc[date]):
                today_scores[ticker] = float(d["score"].loc[date])
        ranking = sorted(today_scores, key=today_scores.get, reverse=True)
        top3    = set(ranking[:N_SLOTS])

        # ── process open positions ───────────────────────────────────────────
        for i, pos in enumerate(slot_pos):
            if pos is None:
                continue

            t      = pos.ticker
            df_t   = stock_data[t]["df"]
            if date not in df_t.index:
                continue

            high  = float(df_t.loc[date, "high"])
            low   = float(df_t.loc[date, "low"])
            close = float(df_t.loc[date, "close"])
            bars_held = bar_idx - pos.entry_bar

            exit_px: Optional[float] = None
            reason  = ""

            # Stop / TP check (intrabar)
            if pos.direction == "long":
                if low <= pos.stop_price:
                    exit_px, reason = pos.stop_price, "stop"
                elif high >= pos.tp_price:
                    exit_px, reason = pos.tp_price,  "tp"
            else:
                if high >= pos.stop_price:
                    exit_px, reason = pos.stop_price, "stop"
                elif low <= pos.tp_price:
                    exit_px, reason = pos.tp_price,  "tp"

            # 3-day orchestrator rule (overrides stop/TP check on same bar)
            if exit_px is None and bars_held <= 3:
                if t not in top3:
                    exit_px, reason = close, "3d-rank"
                else:
                    sig_row = stock_data[t]["sig"].loc[date]
                    opp = (
                        (pos.direction == "long"  and any(bool(sig_row[c]) for c in BEAR_COLS)) or
                        (pos.direction == "short" and any(bool(sig_row[c]) for c in BULL_COLS))
                    )
                    if opp:
                        exit_px, reason = close, "3d-flip"

            if exit_px is not None:
                if pos.direction == "long":
                    pnl_pct = (exit_px - pos.entry_price) / pos.entry_price
                else:
                    pnl_pct = (pos.entry_price - exit_px) / pos.entry_price
                slot_pnl      = slot_cash[i] * pnl_pct
                slot_cash[i] += slot_pnl
                closed_trades.append(ClosedTrade(
                    ticker     = t,
                    direction  = pos.direction,
                    entry_date = pos.entry_date,
                    exit_date  = date,
                    entry_px   = pos.entry_price,
                    exit_px    = exit_px,
                    pnl_pct    = pnl_pct * 100,
                    reason     = reason,
                    slot_pnl   = slot_pnl,
                ))
                slot_pos[i] = None

        # ── fill empty slots ────────────────────────────────────────────────
        held = {p.ticker for p in slot_pos if p is not None}
        for i, pos in enumerate(slot_pos):
            if pos is not None:
                continue
            for ticker in ranking:
                if ticker in held:
                    continue
                sig_row = stock_data[ticker]["sig"]
                if date not in sig_row.index:
                    continue
                row = sig_row.loc[date]
                bull = any(bool(row[c]) for c in BULL_COLS)
                bear = any(bool(row[c]) for c in BEAR_COLS)
                if not bull and not bear:
                    continue

                direction   = "long" if bull else "short"
                entry_price = float(stock_data[ticker]["df"].loc[date, "close"])
                stop_price  = entry_price * (1 - stop_pct/100) if direction == "long" \
                              else entry_price * (1 + stop_pct/100)
                tp_price    = entry_price * (1 + tp_pct/100)   if direction == "long" \
                              else entry_price * (1 - tp_pct/100)

                slot_pos[i] = Position(
                    ticker      = ticker,
                    direction   = direction,
                    entry_date  = date,
                    entry_bar   = bar_idx,
                    entry_price = entry_price,
                    stop_price  = stop_price,
                    tp_price    = tp_price,
                )
                held.add(ticker)
                break

        # ── mark-to-market equity ────────────────────────────────────────────
        total = 0.0
        for i, pos in enumerate(slot_pos):
            if pos is None:
                total += slot_cash[i]
            else:
                t = pos.ticker
                if date in stock_data[t]["df"].index:
                    curr = float(stock_data[t]["df"].loc[date, "close"])
                else:
                    curr = pos.entry_price
                if pos.direction == "long":
                    mtm = slot_cash[i] * (curr / pos.entry_price)
                else:
                    mtm = slot_cash[i] * (2.0 - curr / pos.entry_price)
                total += mtm
        equity_curve.append(total)

    # Close remaining open positions at last bar's close
    last_date = common_dates[-1]
    for i, pos in enumerate(slot_pos):
        if pos is None:
            continue
        t     = pos.ticker
        close = float(stock_data[t]["df"].loc[last_date, "close"])
        if pos.direction == "long":
            pnl_pct = (close - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - close) / pos.entry_price
        slot_pnl      = slot_cash[i] * pnl_pct
        slot_cash[i] += slot_pnl
        closed_trades.append(ClosedTrade(
            ticker=t, direction=pos.direction,
            entry_date=pos.entry_date, exit_date=last_date,
            entry_px=pos.entry_price, exit_px=close,
            pnl_pct=pnl_pct*100, reason="end",
            slot_pnl=slot_pnl,
        ))

    eq = pd.Series(equity_curve, index=common_dates)
    return closed_trades, eq


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    trades: list[ClosedTrade],
    equity: pd.Series,
    initial_capital: float,
) -> None:
    final_val  = equity.iloc[-1]
    total_ret  = (final_val - initial_capital) / initial_capital * 100
    n_years    = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr       = ((final_val / initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    peak  = equity.cummax()
    dd    = (equity - peak) / peak * 100
    max_dd = float(dd.min())

    # Daily returns for Sharpe
    daily_ret = equity.pct_change().dropna()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # Trade stats
    pnls      = [t.pnl_pct for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    win_rate  = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win   = float(np.mean(wins))   if wins   else 0
    avg_loss  = float(np.mean(losses)) if losses else 0
    gross_p   = sum(wins)
    gross_l   = abs(sum(losses))
    pf        = gross_p / gross_l if gross_l > 0 else float("inf")

    exit_reasons = {}
    for t in trades:
        exit_reasons[t.reason] = exit_reasons.get(t.reason, 0) + 1

    print(f"\n{'═'*58}")
    print("  PORTFOLIO SIMULATION REPORT")
    print(f"  {equity.index[0].date()}  →  {equity.index[-1].date()}")
    print(f"{'═'*58}")
    print(f"  Initial capital    : PKR {initial_capital:>12,.0f}")
    print(f"  Final value        : PKR {final_val:>12,.0f}")
    print(f"  Total return       :     {total_ret:>+10.2f}%")
    print(f"  CAGR               :     {cagr:>+10.2f}% p.a.")
    print(f"  Max drawdown       :     {max_dd:>10.2f}%")
    print(f"  Sharpe ratio       :     {sharpe:>10.2f}")
    print(f"{'─'*58}")
    print(f"  Total trades       : {len(trades)}")
    print(f"  Win rate           :     {win_rate:>10.1f}%")
    print(f"  Avg win            :     {avg_win:>+10.2f}%")
    print(f"  Avg loss           :     {avg_loss:>+10.2f}%")
    print(f"  Profit factor      :     {pf:>10.2f}")
    print(f"{'─'*58}")
    print("  Exit reasons:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
        label = {"stop":"Stop-loss","tp":"Take-profit","3d-rank":"3-day rank drop",
                 "3d-flip":"3-day signal flip","end":"End-of-period"}.get(reason, reason)
        print(f"    {label:25s}: {count}")

    # Yearly breakdown
    print(f"\n{'─'*58}")
    print("  Yearly performance:")
    print(f"  {'Year':>4}  {'Start':>10}  {'End':>10}  {'Return':>8}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}")
    years = equity.groupby(equity.index.year)
    for yr, grp in years:
        yr_start = grp.iloc[0]
        yr_end   = grp.iloc[-1]
        yr_ret   = (yr_end - yr_start) / yr_start * 100
        print(f"  {yr:>4}  {yr_start:>10,.0f}  {yr_end:>10,.0f}  {yr_ret:>+8.2f}%")

    # Trade list
    print(f"\n{'─'*58}")
    print("  All trades:")
    hdr = f"  {'#':>3}  {'Ticker':8}  {'Dir':5}  {'Entry':>10}  {'@Px':>8}  {'Exit':>10}  {'@Px':>8}  {'P&L%':>7}  {'PKR P&L':>10}  Reason"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i, t in enumerate(trades, 1):
        entry_d = t.entry_date.date() if hasattr(t.entry_date, "date") else t.entry_date
        exit_d  = t.exit_date.date()  if hasattr(t.exit_date,  "date") else t.exit_date
        print(f"  {i:>3}  {t.ticker:8}  {t.direction:5}  {str(entry_d):>10}  "
              f"{t.entry_px:>8.2f}  {str(exit_d):>10}  {t.exit_px:>8.2f}  "
              f"{t.pnl_pct:>+7.2f}%  {t.slot_pnl:>+10,.0f}  {t.reason}")

    # Equity sparkline (ASCII)
    print(f"\n{'─'*58}")
    print("  Portfolio equity curve (PKR):")
    _print_sparkline(equity, width=54)
    print(f"{'═'*58}\n")


def _print_sparkline(series: pd.Series, width: int = 54) -> None:
    """Simple ASCII equity chart."""
    blocks = "▁▂▃▄▅▆▇█"
    vals   = series.resample("ME").last().dropna()
    if len(vals) < 2:
        return
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx != mn else 1
    bar = "  "
    for v in vals:
        idx = int((v - mn) / rng * (len(blocks) - 1))
        bar += blocks[idx]
    print(bar)
    print(f"  {mn:,.0f} ─── {mx:,.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio simulation")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--stop",    type=float, default=2.0)
    parser.add_argument("--tp",      type=float, default=4.0)
    args = parser.parse_args()

    print(f"\n{'═'*58}")
    print(f"  PORTFOLIO SIMULATION")
    print(f"  Capital: PKR {args.capital:,.0f}  |  {N_SLOTS} equal slots")
    print(f"  Stop: {args.stop}%  |  TP: {args.tp}%  |  3-day rule active")
    print(f"{'═'*58}\n")

    stock_data = precompute(args.stop, args.tp)
    trades, equity = simulate(stock_data, args.capital, args.stop, args.tp)
    print_report(trades, equity, args.capital)


if __name__ == "__main__":
    main()
