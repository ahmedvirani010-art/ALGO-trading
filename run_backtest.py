"""
Entry point: load National Foods CSV, run RSI divergence indicator,
then backtest the generated signals and print a report.

Usage:
    python run_backtest.py [--csv data/NATF_price_history.csv]
                          [--stop 2.0] [--tp 4.0]
                          [--trades]   # print individual trade list
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config.rsi_div_config import Config
from indicators.rsi_divergence import compute
from backtest.backtest_engine import run


# ---------------------------------------------------------------------------
# CSV loader — handles Investing.com style headers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip().strip('"').lower() for c in df.columns]

    # Parse date
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")

    # Numeric columns
    for col in ["price", "open", "high", "low"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    # Rename "price" → "close"
    df = df.rename(columns={"price": "close"})

    return df[["open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RSI Divergence backtest")
    parser.add_argument("--csv", default="data/NATF_price_history.csv")
    parser.add_argument("--stop", type=float, default=2.0,
                        help="Stop-loss %% (default 2.0)")
    parser.add_argument("--tp", type=float, default=4.0,
                        help="Take-profit %% (default 4.0)")
    parser.add_argument("--trades", action="store_true",
                        help="Print individual trade list")
    args = parser.parse_args()

    # Load
    print(f"\nLoading : {args.csv}")
    df = load_csv(args.csv)
    print(f"Bars    : {len(df)}  ({df.index[0].date()} → {df.index[-1].date()})")

    # Pick config that fits the available bars.
    # min_bars = max(p_len_macro*2+1, rsi_length+smooth_length+5)
    n = len(df)
    if n < 79:
        # Short dataset — use reduced periods that still respect min_bars
        rsi_len  = min(14, max(3, n // 4))
        smooth_l = min(30, max(3, n // 4))
        p_macro  = min(10, max(2, (n - 1) // 2))
        # ensure: rsi_len + smooth_l + 5 <= n
        while rsi_len + smooth_l + 5 > n and smooth_l > 3:
            smooth_l -= 1
        while rsi_len + smooth_l + 5 > n and rsi_len > 3:
            rsi_len -= 1
        min_d = max(2, n // 10)
        max_d = max(min_d + 2, n - 2)
        cfg = Config(
            rsi_length=rsi_len,
            smooth_length=smooth_l,
            p_len=1,
            p_len_macro=p_macro,
            min_dist=min_d,
            max_dist=max_d,
            min_rsi_diff=2.0,
        )
    else:
        cfg = Config()

    print(f"Config  : RSI={cfg.rsi_length}, smooth={cfg.smooth_type}/{cfg.smooth_length}, "
          f"p_len={cfg.p_len}/{cfg.p_len_macro}, dist={cfg.min_dist}-{cfg.max_dist}\n")
    df_signals = compute(df, cfg)

    # Count signals
    bull_cols = ["micro_bull_regular","micro_bull_hidden","macro_bull_regular","macro_bull_hidden"]
    bear_cols = ["micro_bear_regular","micro_bear_hidden","macro_bear_regular","macro_bear_hidden"]
    for col in bull_cols + bear_cols:
        cnt = int(df_signals[col].sum())
        if cnt:
            print(f"  {col:30s}: {cnt} signals")

    print()

    # Backtest
    result = run(df_signals, stop_loss_pct=args.stop, take_profit_pct=args.tp)
    print(result.summary())

    if args.trades and result.trades:
        print("\nIndividual trades:")
        print(f"  {'#':>3}  {'Dir':5}  {'Entry':>10}  {'Entry Px':>9}  "
              f"{'Exit':>10}  {'Exit Px':>8}  {'P&L%':>7}  Reason")
        print("  " + "-" * 75)
        for i, t in enumerate(result.trades, 1):
            entry_d = t.entry_date.date() if hasattr(t.entry_date, "date") else t.entry_date
            exit_d  = t.exit_date.date()  if hasattr(t.exit_date,  "date") else t.exit_date
            print(f"  {i:>3}  {t.direction:5}  {str(entry_d):>10}  {t.entry_price:>9.2f}  "
                  f"{str(exit_d):>10}  {t.exit_price:>8.2f}  {t.pnl_pct:>+7.2f}%  {t.exit_reason}")

    if result.total_trades == 0:
        print("\nNo trades generated. Dataset may be too short for the default config.")
        print("Try: --stop 3 --tp 6  or reduce --p-len via the indicator CLI.")

    # Dataset quality warning
    if n < 100:
        print(f"\n⚠  NOTE: Only {n} bars available ({df.index[0].date()} to {df.index[-1].date()}).")
        print("   RSI divergence typically needs 200+ bars for statistically meaningful results.")
        print("   Load a larger history CSV to get reliable backtest metrics.")


if __name__ == "__main__":
    main()
