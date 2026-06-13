"""
Orchestrator: scores all stocks on a multi-factor composite, selects the top N
(default 3) with the highest current chances of success, then reports live signals.

Scoring factors (each 0-20, total 0-100):
  1. Trend strength     — ADX (last bar)
  2. Trend direction    — % distance from 200-SMA (magnitude = conviction)
  3. Retracement pot.   — RSI extremity: >80/<20 best, >70/<30 moderate, near 50 worst
  4. Volatility         — ATR as % of price (more room to move = higher score)
  5. Historical quality — Best Sharpe from grid-search, normalised across all stocks

Usage:
    python orchestrator.py [--top 3] [--stop 2.0] [--tp 4.0] [--min-trades 10]
"""

from __future__ import annotations

import argparse
import itertools
from typing import Any

import numpy as np
import pandas as pd

from config.rsi_div_config import Config
from indicators.rsi_divergence import compute, SIGNAL_COLUMNS
from indicators.ta_utils import (
    compute_rsi, compute_atr, compute_adx, compute_sma_distance,
)
from backtest.backtest_engine import run

# ---------------------------------------------------------------------------
# Stock registry
# ---------------------------------------------------------------------------

STOCKS: dict[str, str] = {
    "NATF":    "data/NATF_price_history_full.csv",
    "BAFL":    "data/BAFL_price_history.csv",
    "HUBC":    "data/HUBC_price_history.csv",
    "ITTEHAD": "data/ITTEHAD_price_history.csv",
    "LUCK":    "data/LUCK_price_history.csv",
    "MARI":    "data/MARI_price_history.csv",
}

PARAM_GRID: dict[str, list[Any]] = {
    "smooth_length": [10, 20, 40],
    "p_len":         [1, 2],
    "p_len_macro":   [3, 5, 10],
    "min_dist":      [3, 5],
    "max_dist":      [100, 150],
    "min_rsi_diff":  [1.0, 2.0, 4.0],
}

COMBOS = [
    p for p in (
        dict(zip(PARAM_GRID.keys(), v))
        for v in itertools.product(*PARAM_GRID.values())
    )
    if p["max_dist"] > p["min_dist"]
]

BULL_SIGNALS = [c for c in SIGNAL_COLUMNS if "bull" in c]
BEAR_SIGNALS = [c for c in SIGNAL_COLUMNS if "bear" in c]


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date").reset_index(drop=True).set_index("date")
    for col in ["price", "open", "high", "low"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return df.rename(columns={"price": "close"})[["open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Historical quality: grid-search best Sharpe
# ---------------------------------------------------------------------------

def best_sharpe(df: pd.DataFrame, stop: float, tp: float, min_trades: int) -> tuple[float, dict]:
    """Return (best_sharpe, best_params_dict) from full grid search."""
    best: float = -999.0
    best_params: dict = COMBOS[0]
    for params in COMBOS:
        try:
            cfg = Config(**params)
            df_sig = compute(df, cfg)
        except ValueError:
            continue
        result = run(df_sig, stop_loss_pct=stop, take_profit_pct=tp)
        if result.total_trades < min_trades:
            continue
        if result.sharpe_ratio > best:
            best = result.sharpe_ratio
            best_params = params
    return best, best_params


# ---------------------------------------------------------------------------
# Sub-scores  (each returns 0-20)
# ---------------------------------------------------------------------------

def score_adx(adx_val: float) -> int:
    if adx_val > 40: return 20
    if adx_val > 30: return 15
    if adx_val > 25: return 10
    if adx_val > 20: return 5
    return 0


def score_sma_dist(dist_pct: float) -> int:
    """Reward strong directional conviction; penalise price hugging the SMA."""
    abs_d = abs(dist_pct)
    if abs_d > 20: return 20
    if abs_d > 10: return 15
    if abs_d > 5:  return 10
    if abs_d > 2:  return 5
    return 0


def score_rsi(rsi_val: float) -> int:
    """RSI 80/20 thresholds signal high retracement potential."""
    dist = abs(rsi_val - 50.0)
    if dist >= 30: return 20   # RSI ≥80 or ≤20
    if dist >= 20: return 10   # RSI ≥70 or ≤30
    if dist >= 10: return 5
    return 0


def score_atr_pct(atr_pct: float) -> int:
    """ATR as % of price — higher volatility = more opportunity."""
    if atr_pct > 3.0: return 20
    if atr_pct > 2.0: return 15
    if atr_pct > 1.5: return 10
    if atr_pct > 1.0: return 5
    return 0


def normalise_sharpe(sharpe: float, all_sharpes: list[float]) -> int:
    """Map best-Sharpe to 0-20 relative to peers."""
    mn, mx = min(all_sharpes), max(all_sharpes)
    if mx == mn:
        return 10
    return int(round((sharpe - mn) / (mx - mn) * 20))


# ---------------------------------------------------------------------------
# Per-stock analysis
# ---------------------------------------------------------------------------

def analyse_stock(
    ticker: str,
    df: pd.DataFrame,
    stop: float,
    tp: float,
    min_trades: int,
) -> dict:
    """Return a dict with raw indicator values, sub-scores, best config."""
    # --- historical quality ---
    sharpe, params = best_sharpe(df, stop, tp, min_trades)

    # --- live indicators (last bar) ---
    adx_df  = compute_adx(df)
    rsi_s   = compute_rsi(df["close"], 14)
    atr_s   = compute_atr(df)
    dist_s  = compute_sma_distance(df["close"], 200)

    last_adx   = float(adx_df["adx"].iloc[-1])
    last_pdi   = float(adx_df["plus_di"].iloc[-1])
    last_mdi   = float(adx_df["minus_di"].iloc[-1])
    last_rsi   = float(rsi_s.iloc[-1])
    last_atr   = float(atr_s.iloc[-1])
    last_close = float(df["close"].iloc[-1])
    last_dist  = float(dist_s.iloc[-1]) if not np.isnan(dist_s.iloc[-1]) else 0.0
    atr_pct    = last_atr / last_close * 100.0

    # --- live signals ---
    try:
        cfg    = Config(**params)
        df_sig = compute(df, cfg)
        last   = df_sig.iloc[-1]
        active_bull = [c for c in BULL_SIGNALS if bool(last.get(c, False))]
        active_bear = [c for c in BEAR_SIGNALS if bool(last.get(c, False))]
    except Exception:
        active_bull, active_bear = [], []

    return {
        "ticker":      ticker,
        "sharpe":      sharpe,
        "best_params": params,
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
        # sub-scores filled in later after normalisation
        "s_adx":  score_adx(last_adx),
        "s_dir":  score_sma_dist(last_dist),
        "s_rsi":  score_rsi(last_rsi),
        "s_atr":  score_atr_pct(atr_pct),
        "s_hist": 0,   # filled after all stocks are analysed
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _rule(widths): return "+-" + "-+-".join("-" * w for w in widths) + "-+"
def _row(cells, widths):
    return "| " + " | ".join(str(c).rjust(w) for c, w in zip(cells, widths)) + " |"


def print_scores(records: list[dict], top_n: int) -> None:
    selected = {r["ticker"] for r in records[:top_n]}
    hdrs  = ["ticker", "ADX", "+DI", "-DI", "RSI", "ATR%", "SMA_d%",
             "s_adx", "s_dir", "s_rsi", "s_atr", "s_hist", "TOTAL", ""]
    widths = [8, 6, 6, 6, 6, 6, 7, 6, 6, 6, 6, 7, 6, 2]
    print(_rule(widths))
    print(_row(hdrs, widths))
    print(_rule(widths))
    for r in records:
        total = r["s_adx"] + r["s_dir"] + r["s_rsi"] + r["s_atr"] + r["s_hist"]
        flag  = "◀" if r["ticker"] in selected else ""
        row = [
            r["ticker"],
            f"{r['adx']:.1f}", f"{r['plus_di']:.1f}", f"{r['minus_di']:.1f}",
            f"{r['rsi']:.1f}", f"{r['atr_pct']:.2f}", f"{r['sma_dist']:+.1f}",
            r["s_adx"], r["s_dir"], r["s_rsi"], r["s_atr"], r["s_hist"],
            total, flag,
        ]
        print(_row(row, widths))
    print(_rule(widths))


def print_signals(records: list[dict], top_n: int) -> None:
    print(f"\n{'═'*62}")
    print(f"  TOP {top_n} — CURRENT SIGNALS")
    print(f"{'═'*62}")
    for r in records[:top_n]:
        ticker = r["ticker"]
        date   = r["last_date"]
        close  = r["last_close"]
        p      = r["best_params"]
        cfg_str = (f"p_len={p['p_len']}, p_macro={p['p_len_macro']}, "
                   f"min_rsi_diff={p['min_rsi_diff']}")

        if r["active_bull"]:
            sig_type = ", ".join(r["active_bull"])
            direction = "LONG "
        elif r["active_bear"]:
            sig_type = ", ".join(r["active_bear"])
            direction = "SHORT"
        else:
            sig_type  = "—"
            direction = "NO SIGNAL"

        print(f"\n  {ticker:8s}  ({date}  close={close:.2f})")
        print(f"    Signal    : {direction}  {sig_type}")
        if r["active_bull"] or r["active_bear"]:
            print(f"    Config    : {cfg_str}")
        print(f"    ADX={r['adx']:.1f}  RSI={r['rsi']:.1f}  ATR%={r['atr_pct']:.2f}  "
              f"SMA_dist={r['sma_dist']:+.1f}%  Sharpe={r['sharpe']:.2f}")
    print(f"\n{'═'*62}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RSI Divergence Orchestrator")
    parser.add_argument("--top",        type=int,   default=3)
    parser.add_argument("--stop",       type=float, default=2.0)
    parser.add_argument("--tp",         type=float, default=4.0)
    parser.add_argument("--min-trades", type=int,   default=10)
    args = parser.parse_args()

    print(f"\n{'═'*62}")
    print(f"  ORCHESTRATOR — Multi-Factor Stock Scoring")
    print(f"  Stop={args.stop}%  TP={args.tp}%  min_trades={args.min_trades}")
    print(f"{'═'*62}\n")

    records: list[dict] = []
    for ticker, csv_path in STOCKS.items():
        print(f"  Analysing {ticker} ...", end="", flush=True)
        try:
            df  = load_csv(csv_path)
            rec = analyse_stock(ticker, df, args.stop, args.tp, args.min_trades)
            records.append(rec)
            print(f"  done  (Sharpe={rec['sharpe']:.2f}  ADX={rec['adx']:.1f}"
                  f"  RSI={rec['rsi']:.1f})")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not records:
        print("No stocks could be analysed.")
        return

    # Normalise historical quality score across peers
    all_sharpes = [r["sharpe"] for r in records]
    for r in records:
        r["s_hist"] = normalise_sharpe(r["sharpe"], all_sharpes)

    # Sort by total composite score
    records.sort(
        key=lambda r: r["s_adx"] + r["s_dir"] + r["s_rsi"] + r["s_atr"] + r["s_hist"],
        reverse=True,
    )

    print(f"\n{'═'*62}")
    print("  SCORING TABLE")
    print(f"{'═'*62}")
    print_scores(records, args.top)
    print_signals(records, args.top)


if __name__ == "__main__":
    main()
