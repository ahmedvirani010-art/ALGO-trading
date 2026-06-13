"""
Multi-stock analysis: run the full parameter grid across every scrip and
produce a per-stock best-config report plus a cross-stock comparison table.

Usage:
    python analyze_all.py [--stop 2.0] [--tp 4.0] [--min-trades 10]
                          [--sort sharpe|profit_factor|total_return|win_rate]
                          [--top 5]
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import time
from typing import Any

import pandas as pd

from config.rsi_div_config import Config
from indicators.rsi_divergence import compute
from backtest.backtest_engine import run

# ---------------------------------------------------------------------------
# Stocks to analyse
# ---------------------------------------------------------------------------

STOCKS = {
    "NATF":    "data/NATF_price_history_full.csv",
    "BAFL":    "data/BAFL_price_history.csv",
    "HUBC":    "data/HUBC_price_history.csv",
    "ITTEHAD": "data/ITTEHAD_price_history.csv",
    "LUCK":    "data/LUCK_price_history.csv",
    "MARI":    "data/MARI_price_history.csv",
}

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

PARAM_GRID: dict[str, list[Any]] = {
    "smooth_length": [10, 20, 40],
    "p_len":         [1, 2],
    "p_len_macro":   [3, 5, 10],
    "min_dist":      [3, 5],
    "max_dist":      [100, 150],
    "min_rsi_diff":  [1.0, 2.0, 4.0],
}

COMBOS = [
    dict(zip(PARAM_GRID.keys(), v))
    for v in itertools.product(*PARAM_GRID.values())
    if list(itertools.product(*PARAM_GRID.values()))[0] or True  # always include
]
# Filter invalid up front
COMBOS = [
    p for p in (
        dict(zip(PARAM_GRID.keys(), v))
        for v in itertools.product(*PARAM_GRID.values())
    )
    if p["max_dist"] > p["min_dist"]
]

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
# Grid search for one stock
# ---------------------------------------------------------------------------

def grid_search_stock(
    ticker: str,
    df: pd.DataFrame,
    stop_loss_pct: float,
    take_profit_pct: float,
    min_trades: int,
) -> list[dict]:
    results = []
    for params in COMBOS:
        try:
            cfg = Config(**params)
            df_sig = compute(df, cfg)
        except ValueError:
            continue
        result = run(df_sig, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
        if result.total_trades < min_trades:
            continue
        row = {"ticker": ticker, **params,
               "trades":        result.total_trades,
               "win_rate":      round(result.win_rate, 1),
               "profit_factor": round(result.profit_factor, 2),
               "total_return":  round(result.total_return_pct, 2),
               "max_drawdown":  round(result.max_drawdown_pct, 2),
               "sharpe":        round(result.sharpe_ratio, 2)}
        results.append(row)
    return results

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

PARAM_COLS  = list(PARAM_GRID.keys())
METRIC_COLS = ["trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "sharpe"]

def _rule(widths): return "+-" + "-+-".join("-" * w for w in widths) + "-+"
def _row(cells, widths):
    return "| " + " | ".join(str(c).rjust(w) for c, w in zip(cells, widths)) + " |"

def print_stock_table(ticker: str, rows: list[dict], sort_key: str, top_n: int) -> None:
    if not rows:
        print(f"  {ticker}: no combinations met the minimum-trades threshold.\n")
        return
    df = pd.DataFrame(rows).sort_values(sort_key, ascending=False).head(top_n)
    cols  = PARAM_COLS + METRIC_COLS
    hdrs  = ["smth_l","p_len","p_macro","min_d","max_d","min_rsi","trades","win%","pf","ret%","dd%","sharpe"]
    widths = [6,5,7,5,5,7,6,5,5,7,7,6]

    print(f"\n{'─'*70}")
    print(f"  {ticker}  — top {min(top_n, len(df))} by {sort_key}")
    print(f"{'─'*70}")
    print(_rule(widths))
    print(_row(hdrs, widths))
    print(_rule(widths))
    for _, r in df.iterrows():
        vals = [r["smooth_length"], r["p_len"], r["p_len_macro"], r["min_dist"],
                r["max_dist"], r["min_rsi_diff"], r["trades"],
                f"{r['win_rate']:.1f}", f"{r['profit_factor']:.2f}",
                f"{r['total_return']:+.1f}", f"{r['max_drawdown']:+.1f}",
                f"{r['sharpe']:.2f}"]
        print(_row(vals, widths))
    print(_rule(widths))

# ---------------------------------------------------------------------------
# Cross-stock summary (one row per ticker = best combo for that ticker)
# ---------------------------------------------------------------------------

def print_summary(summary_rows: list[dict], sort_key: str) -> None:
    if not summary_rows:
        return
    df = pd.DataFrame(summary_rows).sort_values(sort_key, ascending=False)
    cols   = ["ticker","smooth_length","p_len","p_len_macro","min_dist","max_dist",
              "min_rsi_diff","trades","win_rate","profit_factor","total_return","max_drawdown","sharpe"]
    hdrs   = ["ticker","smth_l","p_len","p_macro","min_d","max_d","min_rsi",
              "trades","win%","pf","ret%","dd%","sharpe"]
    widths = [8,6,5,7,5,5,7,6,5,5,7,7,6]

    print(f"\n{'═'*80}")
    print(f"  CROSS-STOCK SUMMARY  (best config per ticker, ranked by {sort_key})")
    print(f"{'═'*80}")
    print(_rule(widths))
    print(_row(hdrs, widths))
    print(_rule(widths))
    for _, r in df.iterrows():
        vals = [r["ticker"], r["smooth_length"], r["p_len"], r["p_len_macro"],
                r["min_dist"], r["max_dist"], r["min_rsi_diff"], r["trades"],
                f"{r['win_rate']:.1f}", f"{r['profit_factor']:.2f}",
                f"{r['total_return']:+.1f}", f"{r['max_drawdown']:+.1f}",
                f"{r['sharpe']:.2f}"]
        print(_row(vals, widths))
    print(_rule(widths))

    # Universal best (highest average sharpe across all stocks)
    print(f"\n{'─'*80}")
    print("  UNIVERSAL BEST CONFIG  (highest mean Sharpe across all stocks)")
    print(f"{'─'*80}")
    all_rows = []
    for r in summary_rows:
        all_rows.append({k: r[k] for k in PARAM_COLS + ["sharpe", "profit_factor",
                                                          "total_return", "win_rate"]})

    # Aggregate by param combo
    param_key = lambda d: tuple(d[k] for k in PARAM_COLS)
    from collections import defaultdict
    agg: dict = defaultdict(list)
    for r in all_rows:
        agg[param_key(r)].append(r["sharpe"])

    best_key = max(agg, key=lambda k: sum(agg[k]) / len(agg[k]))
    best_params = dict(zip(PARAM_COLS, best_key))
    mean_sharpe = sum(agg[best_key]) / len(agg[best_key])
    print("  Parameters:")
    for k, v in best_params.items():
        print(f"    {k:20s} = {v}")
    print(f"  Mean Sharpe across {len(agg[best_key])} stocks: {mean_sharpe:.2f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-stock RSI divergence analysis")
    parser.add_argument("--stop",       type=float, default=2.0)
    parser.add_argument("--tp",         type=float, default=4.0)
    parser.add_argument("--min-trades", type=int,   default=10)
    parser.add_argument("--sort",
                        choices=["sharpe","profit_factor","total_return","win_rate"],
                        default="sharpe")
    parser.add_argument("--top",        type=int,   default=5,
                        help="Top N configs to show per stock (default 5)")
    args = parser.parse_args()

    total_combos = len(COMBOS)
    print(f"\nGrid size : {total_combos} combinations per stock")
    print(f"Stocks    : {len(STOCKS)}  ({', '.join(STOCKS)})")
    print(f"Stop/TP   : {args.stop}% / {args.tp}%   min_trades: {args.min_trades}\n")

    summary_rows: list[dict] = []
    t_global = time.time()

    for ticker, csv_path in STOCKS.items():
        print(f"\nProcessing {ticker} ...", end="", flush=True)
        t0 = time.time()
        try:
            df = load_csv(csv_path)
        except Exception as e:
            print(f"  ERROR loading {csv_path}: {e}")
            continue

        rows = grid_search_stock(ticker, df, args.stop, args.tp, args.min_trades)
        elapsed = time.time() - t0
        valid = len(rows)
        print(f"  {valid} valid combos  ({elapsed:.0f}s)")

        print_stock_table(ticker, rows, args.sort, args.top)

        if rows:
            best = max(rows, key=lambda r: r[args.sort])
            summary_rows.append(best)

    print_summary(summary_rows, args.sort)

    total_elapsed = time.time() - t_global
    print(f"\nTotal time: {total_elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
