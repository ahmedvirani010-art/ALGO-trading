"""
Orchestrator: scores all stocks on a multi-factor composite, selects the top-4
eligible for entry and reports live signals with suggested trade levels.

Scoring factors (each 0-20, total 0-100):
  1. Trend strength     — ADX (last bar)
  2. Trend direction    — % distance from 200-SMA (magnitude = conviction)
  3. Retracement pot.   — RSI extremity: >80/<20 best, >70/<30 moderate, near 50 worst
  4. Volatility         — ATR as % of price (more room to move = higher score)
  5. Historical quality — Pre-computed Sharpe from grid-search, normalised across stocks

Portfolio rules (aligned with optimised back-test):
  Entry  : stock must rank in top 4 AND an RSI divergence signal fires
  Hold   : keep as long as stock stays in top 5
  Exit   : stop-loss 2%  |  take-profit 8%  |  rank drops below 5  |  opposite signal

Usage:
    python orchestrator.py [--top 4] [--stop 2.0] [--tp 8.0]
"""

from __future__ import annotations

import argparse

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

# Pre-computed best configs (from grid-search optimisation on adjusted data)
BEST_CONFIGS: dict[str, dict] = {
    "NATF":    {"smooth_length":10,"p_len":1,"p_len_macro":3, "min_dist":5,"max_dist":100,"min_rsi_diff":4.0},
    "BAFL":    {"smooth_length":10,"p_len":1,"p_len_macro":10,"min_dist":5,"max_dist":100,"min_rsi_diff":4.0},
    "HUBC":    {"smooth_length":10,"p_len":1,"p_len_macro":10,"min_dist":5,"max_dist":100,"min_rsi_diff":4.0},
    "ITTEHAD": {"smooth_length":10,"p_len":1,"p_len_macro":5, "min_dist":5,"max_dist":100,"min_rsi_diff":1.0},
    "LUCK":    {"smooth_length":10,"p_len":2,"p_len_macro":5, "min_dist":3,"max_dist":100,"min_rsi_diff":1.0},
    "MARI":    {"smooth_length":10,"p_len":1,"p_len_macro":10,"min_dist":5,"max_dist":100,"min_rsi_diff":4.0},
}

# Pre-computed historical Sharpe ratios (from portfolio optimisation)
HIST_SHARPE: dict[str, float] = {
    "NATF": 4.25, "BAFL": 2.94, "HUBC": 5.12,
    "ITTEHAD": 3.37, "LUCK": 2.94, "MARI": 9.11,
}

BULL_SIGNALS = [c for c in SIGNAL_COLUMNS if "bull" in c]
BEAR_SIGNALS = [c for c in SIGNAL_COLUMNS if "bear" in c]


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
# Sub-scores (each 0-20)
# ---------------------------------------------------------------------------

def score_adx(v: float) -> int:
    if v > 40: return 20
    if v > 30: return 15
    if v > 25: return 10
    if v > 20: return 5
    return 0


def score_sma_dist(v: float) -> int:
    a = abs(v)
    if a > 20: return 20
    if a > 10: return 15
    if a > 5:  return 10
    if a > 2:  return 5
    return 0


def score_rsi(v: float) -> int:
    """RSI 80/20 thresholds — extreme RSI signals retracement potential."""
    d = abs(v - 50.0)
    if d >= 30: return 20   # RSI ≥80 or ≤20
    if d >= 20: return 10   # RSI ≥70 or ≤30
    if d >= 10: return 5
    return 0


def score_atr_pct(v: float) -> int:
    if v > 3.0: return 20
    if v > 2.0: return 15
    if v > 1.5: return 10
    if v > 1.0: return 5
    return 0


def normalise_sharpe(sharpe: float, mn: float, mx: float) -> int:
    if mx == mn:
        return 10
    return int(round((sharpe - mn) / (mx - mn) * 20))


# ---------------------------------------------------------------------------
# Per-stock analysis
# ---------------------------------------------------------------------------

def analyse_stock(ticker: str, df: pd.DataFrame) -> dict:
    """Compute live indicators + RSI divergence signals using pre-computed best config."""
    cfg    = Config(**BEST_CONFIGS[ticker])
    df_sig = compute(df, cfg)

    adx_df  = compute_adx(df)
    rsi_s   = compute_rsi(df["close"], 14)
    atr_s   = compute_atr(df)
    dist_s  = compute_sma_distance(df["close"], 200)

    last_adx   = float(adx_df["adx"].iloc[-1])
    last_pdi   = float(adx_df["plus_di"].iloc[-1])
    last_mdi   = float(adx_df["minus_di"].iloc[-1])
    last_rsi   = float(rsi_s.iloc[-1])
    last_close = float(df["close"].iloc[-1])
    last_dist  = float(dist_s.iloc[-1]) if not np.isnan(dist_s.iloc[-1]) else 0.0
    atr_pct    = float(atr_s.iloc[-1]) / last_close * 100.0

    last = df_sig.iloc[-1]
    active_bull = [c for c in BULL_SIGNALS if bool(last.get(c, False))]
    active_bear = [c for c in BEAR_SIGNALS if bool(last.get(c, False))]

    return {
        "ticker":      ticker,
        "sharpe":      HIST_SHARPE[ticker],
        "adx":         last_adx,
        "plus_di":     last_pdi,
        "minus_di":    last_mdi,
        "rsi":         last_rsi,
        "atr_pct":     atr_pct,
        "sma_dist":    last_dist,
        "last_close":  last_close,
        "last_date":   df.index[-1].date(),
        "active_bull": active_bull,
        "active_bear": active_bear,
        "s_adx":  score_adx(last_adx),
        "s_dir":  score_sma_dist(last_dist),
        "s_rsi":  score_rsi(last_rsi),
        "s_atr":  score_atr_pct(atr_pct),
        "s_hist": 0,   # filled after normalisation across peers
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _rule(widths): return "+-" + "-+-".join("-" * w for w in widths) + "-+"
def _row(cells, widths):
    return "| " + " | ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w)
                              for i, (c, w) in enumerate(zip(cells, widths))) + " |"


def print_scores(records: list[dict], top_entry: int, top_hold: int) -> None:
    hdrs   = ["Ticker", "Close", "ADX", "+DI", "-DI", "RSI", "ATR%",
              "SMA_d%", "s_adx", "s_dir", "s_rsi", "s_atr", "s_hist", "TOTAL", "Action"]
    widths = [8, 9, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 7, 6, 6]
    print(_rule(widths))
    print(_row(hdrs, widths))
    print(_rule(widths))
    for i, r in enumerate(records):
        total = r["s_adx"] + r["s_dir"] + r["s_rsi"] + r["s_atr"] + r["s_hist"]
        if i < top_entry:
            action = "ENTER"
        elif i < top_hold:
            action = "HOLD"
        else:
            action = "watch"
        row = [
            r["ticker"],
            f"{r['last_close']:,.2f}",
            f"{r['adx']:.1f}", f"{r['plus_di']:.1f}", f"{r['minus_di']:.1f}",
            f"{r['rsi']:.1f}", f"{r['atr_pct']:.2f}", f"{r['sma_dist']:+.1f}",
            r["s_adx"], r["s_dir"], r["s_rsi"], r["s_atr"], r["s_hist"],
            total, action,
        ]
        print(_row(row, widths))
    print(_rule(widths))


def print_signals(records: list[dict], top_entry: int, stop: float, tp: float) -> None:
    print(f"\n{'═'*68}")
    print(f"  ENTRY-ELIGIBLE STOCKS (Top {top_entry})  —  Stop {stop}%  |  TP {tp}%")
    print(f"{'═'*68}")
    for r in records[:top_entry]:
        ticker = r["ticker"]
        close  = r["last_close"]
        date   = r["last_date"]

        if r["active_bull"]:
            sig_type  = ", ".join(r["active_bull"])
            direction = "LONG "
            stop_px   = close * (1 - stop / 100)
            tp_px     = close * (1 + tp   / 100)
        elif r["active_bear"]:
            sig_type  = ", ".join(r["active_bear"])
            direction = "SHORT"
            stop_px   = close * (1 + stop / 100)
            tp_px     = close * (1 - tp   / 100)
        else:
            sig_type  = "—"
            direction = "NO SIGNAL"
            stop_px   = tp_px = None

        p = BEST_CONFIGS[ticker]
        cfg_str = (f"smooth={p['smooth_length']}  p_len={p['p_len']}  "
                   f"p_macro={p['p_len_macro']}  min_rsi_diff={p['min_rsi_diff']}")

        print(f"\n  {ticker:8s}  {date}  close={close:,.2f}")
        print(f"    Signal  : {direction}  {sig_type}")
        if stop_px is not None:
            print(f"    Levels  : entry≈{close:,.2f}  stop={stop_px:,.2f}  TP={tp_px:,.2f}")
        print(f"    Indicators: ADX={r['adx']:.1f}  RSI={r['rsi']:.1f}  "
              f"ATR%={r['atr_pct']:.2f}  SMA_dist={r['sma_dist']:+.1f}%  "
              f"Sharpe(hist)={r['sharpe']:.2f}")
        print(f"    Config  : {cfg_str}")
    print(f"\n{'═'*68}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RSI Divergence Orchestrator (adjusted data)")
    parser.add_argument("--top",       type=int,   default=4,
                        help="Number of stocks eligible for entry (default 4)")
    parser.add_argument("--hold",      type=int,   default=5,
                        help="Hold until rank drops below this (default 5)")
    parser.add_argument("--stop",      type=float, default=2.0)
    parser.add_argument("--tp",        type=float, default=8.0)
    args = parser.parse_args()

    print(f"\n{'═'*68}")
    print(f"  ORCHESTRATOR  —  Multi-Factor Stock Scoring  (adjusted data)")
    print(f"  Entry top-{args.top}  |  Hold top-{args.hold}  |  "
          f"Stop {args.stop}%  |  TP {args.tp}%")
    print(f"{'═'*68}\n")

    records: list[dict] = []
    for ticker, csv_path in STOCKS.items():
        print(f"  {ticker} ...", end="", flush=True)
        try:
            df  = load_csv(csv_path)
            rec = analyse_stock(ticker, df)
            records.append(rec)
            print(f"  ADX={rec['adx']:.1f}  RSI={rec['rsi']:.1f}  "
                  f"close={rec['last_close']:,.2f}  "
                  f"{'BULL' if rec['active_bull'] else 'BEAR' if rec['active_bear'] else '—'}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not records:
        print("No stocks could be analysed.")
        return

    # Normalise historical Sharpe across peers
    sh_vals = [r["sharpe"] for r in records]
    mn, mx  = min(sh_vals), max(sh_vals)
    for r in records:
        r["s_hist"] = normalise_sharpe(r["sharpe"], mn, mx)

    # Sort by composite score
    records.sort(
        key=lambda r: r["s_adx"] + r["s_dir"] + r["s_rsi"] + r["s_atr"] + r["s_hist"],
        reverse=True,
    )

    print(f"\n{'═'*68}")
    print("  SCORING TABLE")
    print(f"{'═'*68}")
    print_scores(records, args.top, args.hold)
    print_signals(records, args.top, args.stop, args.tp)


if __name__ == "__main__":
    main()
