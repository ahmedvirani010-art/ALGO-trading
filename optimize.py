"""
Parameter optimization for the RSI Divergence strategy.

Runs a grid search over key config parameters, backtests every combination,
and prints a ranked results table sorted by Sharpe ratio (or any chosen metric).

Usage:
    python optimize.py [--csv data/NATF_price_history_full.csv]
                       [--stop 2.0] [--tp 4.0]
                       [--sort sharpe|profit_factor|total_return|win_rate]
                       [--top 20]
                       [--min-trades 10]
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
# Parameter grid — edit to widen/narrow the search
# ---------------------------------------------------------------------------

PARAM_GRID: dict[str, list[Any]] = {
    "smooth_length":  [10, 20, 40],
    "p_len":          [1, 2],
    "p_len_macro":    [3, 5, 10],
    "min_dist":       [3, 5],
    "max_dist":       [100, 150],
    "min_rsi_diff":   [1.0, 2.0, 4.0],
}


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
# Grid search
# ---------------------------------------------------------------------------

def grid_search(
    df: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    stop_loss_pct: float,
    take_profit_pct: float,
    min_trades: int,
) -> list[dict]:
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    total = len(combos)
    print(f"Running {total} combinations...\n")

    results = []
    t0 = time.time()

    for i, values in enumerate(combos, 1):
        params = dict(zip(keys, values))

        # Skip invalid combos (max_dist must be > min_dist)
        if params.get("max_dist", 100) <= params.get("min_dist", 5):
            continue

        try:
            cfg = Config(**params)
        except ValueError:
            continue

        try:
            df_sig = compute(df, cfg)
        except ValueError:
            continue

        result = run(df_sig, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)

        if result.total_trades < min_trades:
            continue

        row = {**params}
        row["trades"]         = result.total_trades
        row["win_rate"]       = round(result.win_rate, 1)
        row["avg_win"]        = round(result.avg_win_pct, 2)
        row["avg_loss"]       = round(result.avg_loss_pct, 2)
        row["profit_factor"]  = round(result.profit_factor, 2)
        row["total_return"]   = round(result.total_return_pct, 2)
        row["max_drawdown"]   = round(result.max_drawdown_pct, 2)
        row["sharpe"]         = round(result.sharpe_ratio, 2)
        results.append(row)

        # Progress every 10%
        if i % max(1, total // 10) == 0 or i == total:
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i)
            print(f"  {i:>4}/{total}  done  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    return results


# ---------------------------------------------------------------------------
# Pretty-print results table
# ---------------------------------------------------------------------------

SORT_KEY_MAP = {
    "sharpe":        "sharpe",
    "profit_factor": "profit_factor",
    "total_return":  "total_return",
    "win_rate":      "win_rate",
}

PARAM_COLS = list(PARAM_GRID.keys())
METRIC_COLS = ["trades", "win_rate", "profit_factor", "total_return", "max_drawdown", "sharpe"]

COL_WIDTHS = {
    "smooth_length": 7,
    "p_len":         5,
    "p_len_macro":   9,
    "min_dist":      8,
    "max_dist":      8,
    "min_rsi_diff":  12,
    "trades":        6,
    "win_rate":      8,
    "profit_factor": 13,
    "total_return":  12,
    "max_drawdown":  12,
    "sharpe":        6,
}

HDR_LABELS = {
    "smooth_length": "smth_l",
    "p_len":         "p_len",
    "p_len_macro":   "p_macro",
    "min_dist":      "min_d",
    "max_dist":      "max_d",
    "min_rsi_diff":  "min_rsi_diff",
    "trades":        "trades",
    "win_rate":      "win%",
    "profit_factor": "profit_fac",
    "total_return":  "tot_ret%",
    "max_drawdown":  "max_dd%",
    "sharpe":        "sharpe",
}

ALL_COLS = PARAM_COLS + METRIC_COLS


def _fmt(val: Any, col: str) -> str:
    w = COL_WIDTHS[col]
    if col in ("total_return", "max_drawdown"):
        return f"{val:>+{w}.2f}"
    if col in ("win_rate",):
        return f"{val:>{w}.1f}"
    if col in ("profit_factor", "sharpe"):
        return f"{val:>{w}.2f}"
    return f"{str(val):>{w}}"


def print_table(results: list[dict], sort_key: str, top_n: int) -> None:
    if not results:
        print("No combinations met the minimum-trades threshold.")
        return

    df = pd.DataFrame(results).sort_values(sort_key, ascending=False).head(top_n)

    # Header
    header = "  " + "  ".join(
        f"{HDR_LABELS[c]:>{COL_WIDTHS[c]}}" for c in ALL_COLS
    )
    sep = "  " + "  ".join("-" * COL_WIDTHS[c] for c in ALL_COLS)

    print(f"\nTop {min(top_n, len(df))} results sorted by {sort_key}:\n")
    print(header)
    print(sep)
    for _, row in df.iterrows():
        line = "  " + "  ".join(_fmt(row[c], c) for c in ALL_COLS)
        print(line)
    print()

    # Best overall
    best = df.iloc[0]
    print("Best configuration:")
    for p in PARAM_COLS:
        print(f"  {p:20s} = {best[p]}")
    print()
    print("Performance:")
    for m in METRIC_COLS:
        print(f"  {m:20s} = {best[m]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RSI Divergence parameter optimizer")
    parser.add_argument("--csv",  default="data/NATF_price_history_full.csv")
    parser.add_argument("--stop", type=float, default=2.0)
    parser.add_argument("--tp",   type=float, default=4.0)
    parser.add_argument("--sort", choices=list(SORT_KEY_MAP), default="sharpe",
                        help="Metric to rank by (default: sharpe)")
    parser.add_argument("--top",  type=int, default=20,
                        help="How many top results to display (default: 20)")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Minimum trades to include a result (default: 10)")
    args = parser.parse_args()

    print(f"\nLoading : {args.csv}")
    df = load_csv(args.csv)
    print(f"Bars    : {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")
    print(f"Stop    : {args.stop}%   TP: {args.tp}%   min_trades: {args.min_trades}\n")

    results = grid_search(df, PARAM_GRID, args.stop, args.tp, args.min_trades)

    print(f"\nValid combinations (>= {args.min_trades} trades): {len(results)}")
    sort_col = SORT_KEY_MAP[args.sort]
    print_table(results, sort_col, args.top)


if __name__ == "__main__":
    main()
