"""
Grid search over portfolio simulation parameters to maximise CAGR.

Sweeps:
  n_slots    : 2, 3, 4
  stop_pct   : 1.5, 2.0, 3.0
  tp_pct     : 3.0, 4.0, 6.0, 8.0
  top_entry  : 2, 3, 4
  top_hold   : (top_entry) … 6

Usage:
    python optimize_portfolio.py [--top 20]
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np

from portfolio_sim import precompute, simulate

PARAM_GRID = {
    "n_slots":   [2, 3, 4],
    "stop_pct":  [1.5, 2.0, 3.0],
    "tp_pct":    [3.0, 4.0, 6.0, 8.0],
    "top_entry": [2, 3, 4],
    "top_hold":  [3, 4, 5, 6],
}

INITIAL_CAPITAL = 100_000.0


def _metrics(trades, equity):
    final   = equity.iloc[-1]
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr    = ((final / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    total_r = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    peak   = equity.cummax()
    max_dd = float(((equity - peak) / peak * 100).min())

    dr     = equity.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0

    pnls   = [t.pnl_pct for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else float("inf")

    return {
        "cagr":     cagr,
        "total_r":  total_r,
        "sharpe":   sharpe,
        "max_dd":   max_dd,
        "n_trades": len(trades),
        "win_rt":   len(wins) / len(pnls) * 100 if pnls else 0,
        "pf":       pf,
        "final":    final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    print("Pre-computing signals (once)...")
    stock_data = precompute()
    print()

    combos = [
        dict(zip(PARAM_GRID.keys(), v))
        for v in itertools.product(*PARAM_GRID.values())
        if v[4] >= v[3]   # top_hold >= top_entry
    ]
    print(f"Running {len(combos)} combinations...\n")

    results = []
    for i, p in enumerate(combos, 1):
        trades, equity = simulate(
            stock_data,
            INITIAL_CAPITAL,
            stop_pct=p["stop_pct"],
            tp_pct=p["tp_pct"],
            n_slots=p["n_slots"],
            top_entry=p["top_entry"],
            top_hold=p["top_hold"],
        )
        m = _metrics(trades, equity)
        results.append({**p, **m})
        if i % 50 == 0:
            print(f"  {i}/{len(combos)} done ...")

    results.sort(key=lambda r: r["cagr"], reverse=True)

    print(f"\n{'═'*100}")
    print(f"  TOP {args.top} CONFIGURATIONS  (ranked by CAGR)")
    print(f"{'═'*100}")
    hdr = (f"  {'#':>3}  {'slots':>5}  {'stop%':>5}  {'tp%':>5}  "
           f"{'entry':>5}  {'hold':>4}  {'CAGR':>7}  {'Total%':>7}  "
           f"{'Sharpe':>6}  {'MaxDD':>6}  {'Trades':>6}  {'WinRt':>6}  {'PF':>5}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for i, r in enumerate(results[:args.top], 1):
        print(f"  {i:>3}  {r['n_slots']:>5}  {r['stop_pct']:>5.1f}  {r['tp_pct']:>5.1f}  "
              f"{r['top_entry']:>5}  {r['top_hold']:>4}  "
              f"{r['cagr']:>+7.2f}%  {r['total_r']:>+7.2f}%  "
              f"{r['sharpe']:>6.2f}  {r['max_dd']:>+6.2f}%  "
              f"{r['n_trades']:>6}  {r['win_rt']:>6.1f}%  {r['pf']:>5.2f}")
    print(f"{'═'*100}\n")

    best = results[0]
    print("Best config:")
    print(f"  n_slots={best['n_slots']}  stop={best['stop_pct']}%  tp={best['tp_pct']}%  "
          f"top_entry={best['top_entry']}  top_hold={best['top_hold']}")
    print(f"  CAGR={best['cagr']:+.2f}%  Total={best['total_r']:+.2f}%  "
          f"Sharpe={best['sharpe']:.2f}  MaxDD={best['max_dd']:.2f}%  "
          f"Trades={best['n_trades']}")


if __name__ == "__main__":
    main()
