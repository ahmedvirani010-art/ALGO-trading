"""
Entry point: load a price CSV, run the RSI divergence indicator,
then backtest the generated signals and print a report.

Usage:
    python run_backtest.py [--csv data/NATF_price_history_full.csv]
                          [--preset default|relaxed]
                          [--smooth-length INT] [--p-len INT] [--p-len-macro INT]
                          [--min-dist INT] [--max-dist INT] [--min-rsi-diff FLOAT]
                          [--stop FLOAT] [--tp FLOAT]
                          [--trades]
"""

from __future__ import annotations

import argparse
import dataclasses

import pandas as pd

from config.rsi_div_config import Config, relaxed_config
from indicators.rsi_divergence import compute
from backtest.backtest_engine import run


# ---------------------------------------------------------------------------
# CSV loader — handles Investing.com style headers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"').lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    for col in ["price", "open", "high", "low"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    df = df.rename(columns={"price": "close"})
    return df[["open", "high", "low", "close"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RSI Divergence backtest")
    parser.add_argument("--csv", default="data/NATF_price_history_full.csv")
    parser.add_argument("--preset", choices=["default", "relaxed"], default="relaxed",
                        help="Config preset (default: relaxed)")

    # Per-parameter overrides
    parser.add_argument("--smooth-length", type=int, default=None)
    parser.add_argument("--p-len",         type=int, default=None)
    parser.add_argument("--p-len-macro",   type=int, default=None)
    parser.add_argument("--min-dist",      type=int, default=None)
    parser.add_argument("--max-dist",      type=int, default=None)
    parser.add_argument("--min-rsi-diff",  type=float, default=None)

    # Backtest params
    parser.add_argument("--stop", type=float, default=2.0,
                        help="Stop-loss %% (default 2.0)")
    parser.add_argument("--tp",   type=float, default=4.0,
                        help="Take-profit %% (default 4.0)")
    parser.add_argument("--trades", action="store_true",
                        help="Print individual trade list")
    args = parser.parse_args()

    # Load data
    print(f"\nLoading : {args.csv}")
    df = load_csv(args.csv)
    n = len(df)
    print(f"Bars    : {n}  ({df.index[0].date()} → {df.index[-1].date()})")

    # Build config from preset then apply overrides
    if n < 79:
        # Auto-scale for short datasets
        rsi_len  = min(14, max(3, n // 4))
        smooth_l = min(20, max(3, n // 4))
        p_macro  = min(5,  max(2, (n - 1) // 2))
        while rsi_len + smooth_l + 5 > n and smooth_l > 3:
            smooth_l -= 1
        while rsi_len + smooth_l + 5 > n and rsi_len > 3:
            rsi_len -= 1
        min_d = max(2, n // 10)
        max_d = max(min_d + 2, n - 2)
        cfg = Config(rsi_length=rsi_len, smooth_length=smooth_l, p_len=1,
                     p_len_macro=p_macro, min_dist=min_d, max_dist=max_d,
                     min_rsi_diff=2.0)
    elif args.preset == "relaxed":
        cfg = relaxed_config()
    else:
        cfg = Config()

    # Apply any individual CLI overrides
    overrides: dict = {}
    if args.smooth_length is not None: overrides["smooth_length"] = args.smooth_length
    if args.p_len         is not None: overrides["p_len"]         = args.p_len
    if args.p_len_macro   is not None: overrides["p_len_macro"]   = args.p_len_macro
    if args.min_dist      is not None: overrides["min_dist"]      = args.min_dist
    if args.max_dist      is not None: overrides["max_dist"]      = args.max_dist
    if args.min_rsi_diff  is not None: overrides["min_rsi_diff"]  = args.min_rsi_diff
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)

    preset_label = "auto-scaled" if n < 79 else args.preset
    print(f"Preset  : {preset_label}")
    print(f"Config  : RSI={cfg.rsi_length}, smooth={cfg.smooth_type}/{cfg.smooth_length}, "
          f"p_len={cfg.p_len}/{cfg.p_len_macro}, dist={cfg.min_dist}-{cfg.max_dist}, "
          f"min_rsi_diff={cfg.min_rsi_diff}\n")

    # Generate signals
    df_signals = compute(df, cfg)

    bull_cols = ["micro_bull_regular","micro_bull_hidden","macro_bull_regular","macro_bull_hidden"]
    bear_cols = ["micro_bear_regular","micro_bear_hidden","macro_bear_regular","macro_bear_hidden"]
    total_signals = 0
    for col in bull_cols + bear_cols:
        cnt = int(df_signals[col].sum())
        total_signals += cnt
        if cnt:
            print(f"  {col:30s}: {cnt}")
    print(f"  {'TOTAL signals':30s}: {total_signals}\n")

    # Run backtest
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
        print("\nNo trades generated — try --preset relaxed or lower --min-rsi-diff.")

    if n < 100:
        print(f"\nNOTE: Only {n} bars — load more history for reliable metrics.")


if __name__ == "__main__":
    main()
