"""
Backward price adjustment for dividends.

Algorithm (standard backward-adjustment / "split-adjust" method):
  - Start from the most recent ex-dividend date and work backwards.
  - For each ex-div date D with dividend amount d:
      factor = (unadjusted_close_at_D  -  d) / unadjusted_close_at_D
      All bars BEFORE D get multiplied by this factor (cumulatively).
  - Result: a price series where historical prices are scaled down so that
    the series is continuous — eliminating dividend-driven price drops.

This means total return from any sub-period equals capital gain on the
adjusted series, matching what a buy-and-hold investor actually earned.

Usage (standalone):
    python dividend_adjust.py          # adjusts all stocks, prints summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))
from data.dividend_data import DIVIDENDS

# CSV paths (raw / unadjusted)
RAW_PATHS: dict[str, str] = {
    "NATF":    "data/NATF_price_history_full.csv",
    "BAFL":    "data/BAFL_price_history.csv",
    "HUBC":    "data/HUBC_price_history.csv",
    "ITTEHAD": "data/ITTEHAD_price_history.csv",
    "LUCK":    "data/LUCK_price_history.csv",
    "MARI":    "data/MARI_price_history.csv",
}

ADJ_PATHS: dict[str, str] = {t: p.replace(".csv", "_adj.csv") for t, p in RAW_PATHS.items()}


# ---------------------------------------------------------------------------
# CSV loader (same as portfolio_sim.py)
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
# Core adjustment function
# ---------------------------------------------------------------------------

def adjust_prices(df: pd.DataFrame, dividends: list[tuple[str, float]]) -> pd.DataFrame:
    """
    Return a copy of df with OHLC columns backward-adjusted for dividends.
    Uses the ORIGINAL (unadjusted) close price at each ex-div date to compute
    the factor, so factors do not compound on each other during computation.
    """
    if not dividends:
        return df.copy()

    df_out = df.copy()

    # Sort ex-div dates from newest to oldest
    divs = sorted(
        [(pd.Timestamp(d), amt) for d, amt in dividends],
        key=lambda x: x[0],
        reverse=True,
    )

    for ex_date, div_amt in divs:
        # Nearest trading day at-or-before ex_date (in ORIGINAL data)
        valid = df.index[df.index <= ex_date]
        if len(valid) == 0:
            continue
        actual = valid[-1]

        orig_close = float(df.loc[actual, "close"])   # ORIGINAL price
        if orig_close <= div_amt or orig_close <= 0:
            print(f"    WARNING: div {div_amt:.4f} >= close {orig_close:.4f} on {actual.date()} — skipped")
            continue

        factor = (orig_close - div_amt) / orig_close

        # Apply to all bars strictly before the ex-div date (in df_out)
        mask = df_out.index < actual
        df_out.loc[mask, ["open", "high", "low", "close"]] *= factor

    return df_out


# ---------------------------------------------------------------------------
# Build and save all adjusted CSVs
# ---------------------------------------------------------------------------

def build_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Adjust all stocks and save *_adj.csv files. Returns dict of adjusted dfs."""
    adjusted: dict[str, pd.DataFrame] = {}

    for ticker, raw_path in RAW_PATHS.items():
        divs = DIVIDENDS.get(ticker, [])

        if verbose:
            print(f"\n{ticker}  ({len(divs)} dividend events)")

        df_raw = load_csv(raw_path)
        df_adj = adjust_prices(df_raw, divs)

        # How much did the earliest price change?
        if divs and verbose:
            old_first = df_raw["close"].iloc[0]
            new_first = df_adj["close"].iloc[0]
            print(f"    Earliest close: {old_first:.2f}  →  {new_first:.2f}  "
                  f"(factor {new_first/old_first:.4f})")
            old_last  = df_raw["close"].iloc[-1]
            new_last  = df_adj["close"].iloc[-1]
            print(f"    Latest  close: {old_last:.2f}  →  {new_last:.2f}  "
                  f"(last bar unchanged by design)")

        # Save as CSV (keep original Date column format)
        adj_path = ADJ_PATHS[ticker]
        df_adj.reset_index().rename(columns={"date": "Date", "close": "Price"}
            ).assign(**{"Change %": ""}).to_csv(adj_path, index=False)

        if verbose:
            print(f"    Saved → {adj_path}")

        adjusted[ticker] = df_adj

    return adjusted


# ---------------------------------------------------------------------------
# Main (standalone run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  DIVIDEND PRICE ADJUSTMENT")
    print("=" * 60)
    build_all(verbose=True)
    print("\nDone.")
