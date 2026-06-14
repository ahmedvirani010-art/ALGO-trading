"""
Walk-forward portfolio simulation — Orchestrator-driven, 2-slot version.

Rules
-----
- Capital  : PKR 100,000 split equally into 2 slots (50,000 each)
- Entry    : slot is empty AND orchestrator ranks stock in top 2
             AND an RSI divergence signal fires on that bar
- Hold     : keep position as long as stock stays in orchestrator top 4
- Exit     : (a) stop-loss 2% hit  (intrabar low/high)
             (b) take-profit 4% hit (intrabar high/low)
             (c) stock drops out of orchestrator top-4 → exit at close
             (d) opposite RSI divergence signal fires → exit at close
- Replace  : after any exit, scan current top-2 for next entry signal
- Sizing   : full slot value per trade (compounding)

Data       : dividend-adjusted OHLC  (*_adj.csv files)

Usage:
    python portfolio_sim.py [--capital 100000] [--stop 2.0] [--tp 4.0]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.rsi_div_config import Config
from indicators.rsi_divergence import compute, SIGNAL_COLUMNS
from indicators.ta_utils import (
    compute_rsi, compute_atr, compute_adx, compute_sma_distance,
)

# ---------------------------------------------------------------------------
# Stock registry — dividend-adjusted CSVs
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

HIST_SHARPE: dict[str, float] = {
    "NATF": 4.25, "BAFL": 2.94, "HUBC": 5.12,
    "ITTEHAD": 3.37, "LUCK": 2.94, "MARI": 9.11,
}

N_SLOTS   = 2   # number of concurrent positions
TOP_ENTRY = 2   # must be in top-N to enter
TOP_HOLD  = 4   # exit if rank drops below this

BULL_COLS = [c for c in SIGNAL_COLUMNS if "bull" in c]
BEAR_COLS = [c for c in SIGNAL_COLUMNS if "bear" in c]


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

def _s_adx(v):
    if v>40: return 20
    if v>30: return 15
    if v>25: return 10
    if v>20: return 5
    return 0

def _s_dir(v):
    a=abs(v)
    if a>20: return 20
    if a>10: return 15
    if a>5:  return 10
    if a>2:  return 5
    return 0

def _s_rsi(v):
    d=abs(v-50)
    if d>=30: return 20
    if d>=20: return 10
    if d>=10: return 5
    return 0

def _s_atr(v):
    if v>3.0: return 20
    if v>2.0: return 15
    if v>1.5: return 10
    if v>1.0: return 5
    return 0

def _norm_sh(s, mn, mx):
    return int(round((s-mn)/(mx-mn)*20)) if mx!=mn else 10


# ---------------------------------------------------------------------------
# Precompute per-stock series
# ---------------------------------------------------------------------------

def precompute() -> dict[str, dict]:
    print("Pre-computing signals and indicator scores (adjusted data)...")
    sh_vals = list(HIST_SHARPE.values())
    sh_mn, sh_mx = min(sh_vals), max(sh_vals)

    stock_data: dict[str, dict] = {}
    for ticker, csv_path in STOCKS.items():
        print(f"  {ticker} ...", end="", flush=True)
        df = load_csv(csv_path)

        cfg    = Config(**BEST_CONFIGS[ticker])
        df_sig = compute(df, cfg)

        adx_df  = compute_adx(df)
        rsi_s   = compute_rsi(df["close"], 14)
        atr_s   = compute_atr(df)
        dist_s  = compute_sma_distance(df["close"], 200)
        atr_pct = atr_s / df["close"] * 100.0
        s_hist  = _norm_sh(HIST_SHARPE[ticker], sh_mn, sh_mx)

        score_s = (
            adx_df["adx"].apply(_s_adx)
            + dist_s.apply(_s_dir)
            + rsi_s.apply(_s_rsi)
            + atr_pct.apply(_s_atr)
            + s_hist
        )

        stock_data[ticker] = {
            "df":    df,
            "sig":   df_sig[SIGNAL_COLUMNS],
            "score": score_s,
        }
        print(" done")
    return stock_data


# ---------------------------------------------------------------------------
# Position & trade dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker:      str
    direction:   str
    entry_date:  pd.Timestamp
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
    slot_pnl:   float


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    stock_data: dict[str, dict],
    initial_capital: float,
    stop_pct: float,
    tp_pct: float,
) -> tuple[list[ClosedTrade], pd.Series]:

    date_sets    = [set(stock_data[t]["df"].index) for t in stock_data]
    common_dates = sorted(set.intersection(*date_sets))

    slot_cash: list[float]             = [initial_capital / N_SLOTS] * N_SLOTS
    slot_pos:  list[Optional[Position]] = [None] * N_SLOTS

    equity_curve: list[float] = []
    closed_trades: list[ClosedTrade] = []

    for date in common_dates:

        # ── orchestrator ranking ─────────────────────────────────────────
        today_scores = {
            t: float(d["score"].loc[date])
            for t, d in stock_data.items()
            if date in d["score"].index and not np.isnan(d["score"].loc[date])
        }
        ranking = sorted(today_scores, key=today_scores.get, reverse=True)
        top_entry_set = set(ranking[:TOP_ENTRY])
        top_hold_set  = set(ranking[:TOP_HOLD])

        # ── manage open positions ────────────────────────────────────────
        for i, pos in enumerate(slot_pos):
            if pos is None:
                continue
            t     = pos.ticker
            df_t  = stock_data[t]["df"]
            if date not in df_t.index:
                continue

            high  = float(df_t.loc[date, "high"])
            low   = float(df_t.loc[date, "low"])
            close = float(df_t.loc[date, "close"])

            exit_px: Optional[float] = None
            reason  = ""

            # (a) Stop-loss / Take-profit
            if pos.direction == "long":
                if low  <= pos.stop_price: exit_px, reason = pos.stop_price, "stop"
                elif high >= pos.tp_price: exit_px, reason = pos.tp_price,   "tp"
            else:
                if high >= pos.stop_price: exit_px, reason = pos.stop_price, "stop"
                elif low  <= pos.tp_price: exit_px, reason = pos.tp_price,   "tp"

            # (b) Rank drop below top-4
            if exit_px is None and t not in top_hold_set:
                exit_px, reason = close, "rank-drop"

            # (c) Opposite RSI signal
            if exit_px is None:
                sig_row = stock_data[t]["sig"].loc[date]
                if pos.direction == "long"  and any(bool(sig_row[c]) for c in BEAR_COLS):
                    exit_px, reason = close, "opp-signal"
                elif pos.direction == "short" and any(bool(sig_row[c]) for c in BULL_COLS):
                    exit_px, reason = close, "opp-signal"

            if exit_px is not None:
                pnl = (exit_px - pos.entry_price)/pos.entry_price if pos.direction=="long" \
                      else (pos.entry_price - exit_px)/pos.entry_price
                slot_pnl      = slot_cash[i] * pnl
                slot_cash[i] += slot_pnl
                closed_trades.append(ClosedTrade(
                    ticker=t, direction=pos.direction,
                    entry_date=pos.entry_date, exit_date=date,
                    entry_px=pos.entry_price, exit_px=exit_px,
                    pnl_pct=pnl*100, reason=reason, slot_pnl=slot_pnl,
                ))
                slot_pos[i] = None

        # ── fill empty slots ──────────────────────────────────────────────
        held = {p.ticker for p in slot_pos if p is not None}
        for i, pos in enumerate(slot_pos):
            if pos is not None:
                continue
            for ticker in ranking[:TOP_ENTRY]:      # only enter from top-2
                if ticker in held:
                    continue
                if date not in stock_data[ticker]["sig"].index:
                    continue
                row  = stock_data[ticker]["sig"].loc[date]
                bull = any(bool(row[c]) for c in BULL_COLS)
                bear = any(bool(row[c]) for c in BEAR_COLS)
                if not bull and not bear:
                    continue

                direction   = "long" if bull else "short"
                entry_price = float(stock_data[ticker]["df"].loc[date, "close"])
                stop_price  = entry_price*(1-stop_pct/100)  if direction=="long" \
                              else entry_price*(1+stop_pct/100)
                tp_price    = entry_price*(1+tp_pct/100)    if direction=="long" \
                              else entry_price*(1-tp_pct/100)

                slot_pos[i] = Position(
                    ticker=ticker, direction=direction,
                    entry_date=date, entry_price=entry_price,
                    stop_price=stop_price, tp_price=tp_price,
                )
                held.add(ticker)
                break

        # ── mark-to-market ────────────────────────────────────────────────
        total = 0.0
        for i, pos in enumerate(slot_pos):
            if pos is None:
                total += slot_cash[i]
            else:
                t    = pos.ticker
                curr = float(stock_data[t]["df"].loc[date, "close"]) \
                       if date in stock_data[t]["df"].index else pos.entry_price
                mtm  = slot_cash[i] * (curr/pos.entry_price) if pos.direction=="long" \
                       else slot_cash[i] * (2.0 - curr/pos.entry_price)
                total += mtm
        equity_curve.append(total)

    # Close remaining at last bar
    last_date = common_dates[-1]
    for i, pos in enumerate(slot_pos):
        if pos is None:
            continue
        t     = pos.ticker
        close = float(stock_data[t]["df"].loc[last_date, "close"])
        pnl   = (close-pos.entry_price)/pos.entry_price if pos.direction=="long" \
                else (pos.entry_price-close)/pos.entry_price
        slot_pnl      = slot_cash[i] * pnl
        slot_cash[i] += slot_pnl
        closed_trades.append(ClosedTrade(
            ticker=pos.ticker, direction=pos.direction,
            entry_date=pos.entry_date, exit_date=last_date,
            entry_px=pos.entry_price, exit_px=close,
            pnl_pct=pnl*100, reason="end", slot_pnl=slot_pnl,
        ))

    return closed_trades, pd.Series(equity_curve, index=common_dates)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(trades: list[ClosedTrade], equity: pd.Series, initial_capital: float) -> None:
    final_val = equity.iloc[-1]
    total_ret = (final_val - initial_capital) / initial_capital * 100
    n_years   = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr      = ((final_val/initial_capital)**(1/n_years)-1)*100 if n_years>0 else 0

    peak   = equity.cummax()
    max_dd = float(((equity - peak)/peak*100).min())

    daily_ret = equity.pct_change().dropna()
    sharpe    = float(daily_ret.mean()/daily_ret.std()*np.sqrt(252)) if daily_ret.std()>0 else 0

    pnls    = [t.pnl_pct for t in trades]
    wins    = [p for p in pnls if p>0]
    losses  = [p for p in pnls if p<=0]
    win_rt  = len(wins)/len(pnls)*100 if pnls else 0
    avg_w   = float(np.mean(wins))   if wins   else 0
    avg_l   = float(np.mean(losses)) if losses else 0
    pf      = sum(wins)/abs(sum(losses)) if losses else float("inf")

    reasons: dict[str,int] = {}
    for t in trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    print(f"\n{'═'*60}")
    print("  PORTFOLIO REPORT  (Orchestrator Top-2, Hold to Top-4)")
    print(f"  {equity.index[0].date()}  →  {equity.index[-1].date()}")
    print(f"{'═'*60}")
    print(f"  Initial capital    : PKR {initial_capital:>12,.0f}")
    print(f"  Final value        : PKR {final_val:>12,.0f}")
    print(f"  Total return       :     {total_ret:>+10.2f}%")
    print(f"  CAGR               :     {cagr:>+10.2f}% p.a.")
    print(f"  Max drawdown       :     {max_dd:>10.2f}%")
    print(f"  Sharpe ratio       :     {sharpe:>10.2f}")
    print(f"{'─'*60}")
    print(f"  Total trades       : {len(trades)}")
    print(f"  Win rate           :     {win_rt:>10.1f}%")
    print(f"  Avg win            :     {avg_w:>+10.2f}%")
    print(f"  Avg loss           :     {avg_l:>+10.2f}%")
    print(f"  Profit factor      :     {pf:>10.2f}")
    print(f"{'─'*60}")
    print("  Exit reasons:")
    labels = {"stop":"Stop-loss","tp":"Take-profit","rank-drop":"Rank drop (out of top-4)",
              "opp-signal":"Opposite signal","end":"End-of-period"}
    for r, n in sorted(reasons.items(), key=lambda x:-x[1]):
        print(f"    {labels.get(r,r):30s}: {n}")

    # Yearly
    print(f"\n{'─'*60}")
    print("  Yearly performance:")
    print(f"  {'Year':>4}  {'Start':>10}  {'End':>10}  {'Return':>8}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}")
    for yr, grp in equity.groupby(equity.index.year):
        s, e = grp.iloc[0], grp.iloc[-1]
        print(f"  {yr:>4}  {s:>10,.0f}  {e:>10,.0f}  {(e-s)/s*100:>+8.2f}%")

    # Trade list
    print(f"\n{'─'*60}")
    print("  All trades:")
    hdr = (f"  {'#':>3}  {'Ticker':8}  {'Dir':5}  {'Entry':>10}  {'@Px':>8}  "
           f"{'Exit':>10}  {'@Px':>8}  {'P&L%':>7}  {'PKR P&L':>10}  Reason")
    print(hdr)
    print("  " + "─"*(len(hdr)-2))
    for i, t in enumerate(trades, 1):
        ed = t.entry_date.date() if hasattr(t.entry_date,"date") else t.entry_date
        xd = t.exit_date.date()  if hasattr(t.exit_date, "date") else t.exit_date
        print(f"  {i:>3}  {t.ticker:8}  {t.direction:5}  {str(ed):>10}  "
              f"{t.entry_px:>8.2f}  {str(xd):>10}  {t.exit_px:>8.2f}  "
              f"{t.pnl_pct:>+7.2f}%  {t.slot_pnl:>+10,.0f}  {t.reason}")

    # Sparkline
    print(f"\n{'─'*60}")
    print("  Portfolio equity curve (PKR):")
    blocks = "▁▂▃▄▅▆▇█"
    vals   = equity.resample("ME").last().dropna()
    mn, mx = vals.min(), vals.max()
    rng    = mx - mn if mx != mn else 1
    bar    = "  " + "".join(blocks[int((v-mn)/rng*(len(blocks)-1))] for v in vals)
    print(bar)
    print(f"  {mn:,.0f} ─── {mx:,.0f}")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator 2-slot portfolio simulation")
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--stop",    type=float, default=2.0)
    parser.add_argument("--tp",      type=float, default=4.0)
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  ORCHESTRATOR PORTFOLIO  |  2 slots  |  Top-2 entry / Top-4 hold")
    print(f"  Capital: PKR {args.capital:,.0f}  |  Stop: {args.stop}%  |  TP: {args.tp}%")
    print(f"  Data: dividend-adjusted prices")
    print(f"{'═'*60}\n")

    stock_data = precompute()
    trades, equity = simulate(stock_data, args.capital, args.stop, args.tp)
    print_report(trades, equity, args.capital)


if __name__ == "__main__":
    main()
