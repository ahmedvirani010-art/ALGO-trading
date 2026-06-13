"""
Micro and Macro RSI Divergence — Python implementation.

Translated from Pine Script v6 indicator "Micro and Macro RSI Divergence"
by Uncle_the_shooter (MPL 2.0).

Usage:
    from indicators.rsi_divergence import compute
    from config import Config

    result = compute(df, Config())
    # result has 8 extra boolean columns:
    #   micro_bull_regular, micro_bear_regular
    #   micro_bull_hidden,  micro_bear_hidden
    #   macro_bull_regular, macro_bear_regular
    #   macro_bull_hidden,  macro_bear_hidden
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from config.rsi_div_config import Config
from indicators.ta_utils import (
    compute_rsi,
    compute_source,
    pivot_high,
    pivot_low,
    smooth_rsi,
    valuewhen,
)

SIGNAL_COLUMNS = [
    "micro_bull_regular",
    "micro_bear_regular",
    "micro_bull_hidden",
    "micro_bear_hidden",
    "macro_bull_regular",
    "macro_bear_regular",
    "macro_bull_hidden",
    "macro_bear_hidden",
]


# ---------------------------------------------------------------------------
# Line-break filter
# ---------------------------------------------------------------------------

def _line_is_broken(
    src_arr: np.ndarray,
    rsi_arr: np.ndarray,
    bar_a: int,
    bar_b: int,
    price_a: float,
    price_b: float,
    rsi_a: float,
    rsi_b: float,
    direction: str,  # "low" or "high"
) -> bool:
    """
    Return True if ANY intermediate bar breaks the straight line connecting
    the two pivots — checking both the price line and the RSI line.

    direction="low"  → bull divergence; line broken if intermediate goes BELOW
    direction="high" → bear divergence; line broken if intermediate goes ABOVE
    """
    if bar_b <= bar_a + 1:
        return False

    dist = bar_b - bar_a
    t = np.arange(1, dist, dtype=float)
    frac = t / dist

    line_price = price_a + frac * (price_b - price_a)
    line_rsi = rsi_a + frac * (rsi_b - rsi_a)

    price_slice = src_arr[bar_a + 1: bar_b]
    rsi_slice = rsi_arr[bar_a + 1: bar_b]

    if direction == "low":
        return bool(np.any(price_slice < line_price) or np.any(rsi_slice < line_rsi))
    else:
        return bool(np.any(price_slice > line_price) or np.any(rsi_slice > line_rsi))


# ---------------------------------------------------------------------------
# Per-scale divergence computation
# ---------------------------------------------------------------------------

def _compute_scale(
    *,
    pivot_low_price: pd.Series,
    pivot_high_price: pd.Series,
    low_confirmed: pd.Series,
    high_confirmed: pd.Series,
    rsi_at_low: pd.Series,
    rsi_at_high: pd.Series,
    srsi_at_low: pd.Series,
    srsi_at_high: pd.Series,
    prev_low_price: pd.Series,
    prev_high_price: pd.Series,
    prev_low_rsi: pd.Series,
    prev_high_rsi: pd.Series,
    prev_low_srsi: pd.Series,
    prev_high_srsi: pd.Series,
    dist_ok_low: pd.Series,
    dist_ok_high: pd.Series,
    prev_low_bar: pd.Series,
    prev_high_bar: pd.Series,
    bar_positions: pd.Series,
    price_low_arr: np.ndarray,
    price_high_arr: np.ndarray,
    rsi_arr: np.ndarray,
    p_len: int,
    config: Config,
) -> dict[str, pd.Series]:
    """Compute 4 divergence signals for one pivot scale (micro or macro)."""

    n = len(pivot_low_price)
    bull_reg = np.zeros(n, dtype=bool)
    bear_reg = np.zeros(n, dtype=bool)
    bull_hid = np.zeros(n, dtype=bool)
    bear_hid = np.zeros(n, dtype=bool)

    pos_arr = bar_positions.values
    rsi_vals = rsi_arr

    # ---- LOW PIVOT (bull regular + bull hidden) ----------------------------
    for iloc in range(n):
        if not (low_confirmed.iloc[iloc] and dist_ok_low.iloc[iloc]):
            continue

        cur_price = pivot_low_price.iloc[iloc]
        prv_price = prev_low_price.iloc[iloc]
        cur_rsi = rsi_at_low.iloc[iloc]
        prv_rsi = prev_low_rsi.iloc[iloc]
        cur_srsi = srsi_at_low.iloc[iloc]
        prv_srsi = prev_low_srsi.iloc[iloc]

        if any(pd.isna(v) for v in (cur_price, prv_price, cur_rsi, prv_rsi)):
            continue

        rsi_diff_reg = cur_rsi - prv_rsi
        rsi_diff_hid = prv_rsi - cur_rsi

        # distance between confirmation bars
        bar_b = int(pos_arr[iloc])
        bar_a = int(prev_low_bar.iloc[iloc])
        if pd.isna(bar_a):
            continue
        bar_a = int(bar_a)

        # line break for price-low / rsi arrays
        line_broken = _line_is_broken(
            price_low_arr, rsi_vals,
            bar_a - p_len, bar_b - p_len,  # actual pivot bar positions
            float(prv_price), float(cur_price),
            float(prv_rsi), float(cur_rsi),
            "low",
        )

        # optional filter helpers
        def _low_filters_ok(direction_is_up: bool) -> bool:
            if config.filter_rsi_50 and cur_rsi <= 50:
                return False
            if config.filter_smooth_rsi_50 and not pd.isna(cur_srsi) and cur_srsi <= 50:
                return False
            if config.filter_rsi_direction:
                if direction_is_up and cur_rsi <= prv_rsi:
                    return False
                if not direction_is_up and cur_rsi >= prv_rsi:
                    return False
            if config.filter_smooth_rsi_direction and not pd.isna(cur_srsi) and not pd.isna(prv_srsi):
                if direction_is_up and cur_srsi <= prv_srsi:
                    return False
                if not direction_is_up and cur_srsi >= prv_srsi:
                    return False
            return True

        # Regular bull: price lower low, RSI higher low
        if (
            cur_price < prv_price
            and cur_rsi > prv_rsi
            and rsi_diff_reg >= config.min_rsi_diff
            and not line_broken
            and _low_filters_ok(True)
        ):
            bull_reg[iloc] = True

        # Hidden bull: price higher low, RSI lower low
        if (
            cur_price > prv_price
            and cur_rsi < prv_rsi
            and rsi_diff_hid >= config.min_rsi_diff
            and not line_broken
            and _low_filters_ok(False)
        ):
            bull_hid[iloc] = True

    # ---- HIGH PIVOT (bear regular + bear hidden) ---------------------------
    for iloc in range(n):
        if not (high_confirmed.iloc[iloc] and dist_ok_high.iloc[iloc]):
            continue

        cur_price = pivot_high_price.iloc[iloc]
        prv_price = prev_high_price.iloc[iloc]
        cur_rsi = rsi_at_high.iloc[iloc]
        prv_rsi = prev_high_rsi.iloc[iloc]
        cur_srsi = srsi_at_high.iloc[iloc]
        prv_srsi = prev_high_srsi.iloc[iloc]

        if any(pd.isna(v) for v in (cur_price, prv_price, cur_rsi, prv_rsi)):
            continue

        rsi_diff_reg = prv_rsi - cur_rsi
        rsi_diff_hid = cur_rsi - prv_rsi

        bar_b = int(pos_arr[iloc])
        bar_a = int(prev_high_bar.iloc[iloc])
        if pd.isna(bar_a):
            continue
        bar_a = int(bar_a)

        line_broken = _line_is_broken(
            price_high_arr, rsi_vals,
            bar_a - p_len, bar_b - p_len,
            float(prv_price), float(cur_price),
            float(prv_rsi), float(cur_rsi),
            "high",
        )

        def _high_filters_ok(direction_is_down: bool) -> bool:
            if config.filter_rsi_50 and cur_rsi >= 50:
                return False
            if config.filter_smooth_rsi_50 and not pd.isna(cur_srsi) and cur_srsi >= 50:
                return False
            if config.filter_rsi_direction:
                if direction_is_down and cur_rsi >= prv_rsi:
                    return False
                if not direction_is_down and cur_rsi <= prv_rsi:
                    return False
            if config.filter_smooth_rsi_direction and not pd.isna(cur_srsi) and not pd.isna(prv_srsi):
                if direction_is_down and cur_srsi >= prv_srsi:
                    return False
                if not direction_is_down and cur_srsi <= prv_srsi:
                    return False
            return True

        # Regular bear: price higher high, RSI lower high
        if (
            cur_price > prv_price
            and cur_rsi < prv_rsi
            and rsi_diff_reg >= config.min_rsi_diff
            and not line_broken
            and _high_filters_ok(True)
        ):
            bear_reg[iloc] = True

        # Hidden bear: price lower high, RSI higher high
        if (
            cur_price < prv_price
            and cur_rsi > prv_rsi
            and rsi_diff_hid >= config.min_rsi_diff
            and not line_broken
            and _high_filters_ok(False)
        ):
            bear_hid[iloc] = True

    idx = pivot_low_price.index
    return {
        "bull_regular": pd.Series(bull_reg, index=idx),
        "bear_regular": pd.Series(bear_reg, index=idx),
        "bull_hidden": pd.Series(bull_hid, index=idx),
        "bear_hidden": pd.Series(bear_hid, index=idx),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute(df: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """
    Compute Micro and Macro RSI Divergence signals.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns ['open','high','low','close'] (volume optional).
        Index must be monotonically increasing.
    config : Config, optional
        Indicator configuration.  Defaults to Config() (Pine Script defaults).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with 8 boolean signal columns appended.
    """
    if config is None:
        config = Config()

    # --- validation ---------------------------------------------------------
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be monotonically increasing")

    min_bars = max(
        config.p_len_macro * 2 + 1,
        config.rsi_length + config.smooth_length + 5,
    )
    if len(df) < min_bars:
        raise ValueError(f"Need at least {min_bars} bars; got {len(df)}")

    # --- core computations --------------------------------------------------
    rsi_src = compute_source(df, config.rsi_source)
    rsi = compute_rsi(rsi_src, config.rsi_length)
    srsi = smooth_rsi(rsi, config.smooth_type, config.smooth_length)

    if config.pivot_source == "hl":
        piv_low_src = df["low"]
        piv_high_src = df["high"]
    else:
        piv_low_src = df["close"]
        piv_high_src = df["close"]

    bar_positions = pd.Series(np.arange(len(df)), index=df.index, dtype=float)

    price_low_arr = piv_low_src.values.astype(float)
    price_high_arr = piv_high_src.values.astype(float)
    rsi_arr = rsi.values.astype(float)

    result = df.copy()

    for scale, p_len in [("micro", config.p_len), ("macro", config.p_len_macro)]:
        piv_l = pivot_low(piv_low_src, p_len, p_len)
        piv_h = pivot_high(piv_high_src, p_len, p_len)

        low_confirmed = piv_l.notna()
        high_confirmed = piv_h.notna()

        # RSI value at the actual pivot bar (p_len bars before confirmation)
        rsi_at_low = rsi.shift(p_len).where(low_confirmed)
        rsi_at_high = rsi.shift(p_len).where(high_confirmed)
        srsi_at_low = srsi.shift(p_len).where(low_confirmed)
        srsi_at_high = srsi.shift(p_len).where(high_confirmed)

        # previous pivot values
        prev_low_price = valuewhen(low_confirmed, piv_l, 1)
        prev_high_price = valuewhen(high_confirmed, piv_h, 1)
        prev_low_rsi = valuewhen(low_confirmed, rsi_at_low, 1)
        prev_high_rsi = valuewhen(high_confirmed, rsi_at_high, 1)
        prev_low_srsi = valuewhen(low_confirmed, srsi_at_low, 1)
        prev_high_srsi = valuewhen(high_confirmed, srsi_at_high, 1)

        # distance from this confirmation bar to the previous pivot confirmation bar
        prev_low_bar = valuewhen(low_confirmed, bar_positions, 1)
        prev_high_bar = valuewhen(high_confirmed, bar_positions, 1)

        dist_low = (bar_positions - prev_low_bar).where(low_confirmed)
        dist_high = (bar_positions - prev_high_bar).where(high_confirmed)

        dist_ok_low = low_confirmed & (dist_low >= config.min_dist) & (dist_low <= config.max_dist)
        dist_ok_high = high_confirmed & (dist_high >= config.min_dist) & (dist_high <= config.max_dist)

        sigs = _compute_scale(
            pivot_low_price=piv_l,
            pivot_high_price=piv_h,
            low_confirmed=low_confirmed,
            high_confirmed=high_confirmed,
            rsi_at_low=rsi_at_low,
            rsi_at_high=rsi_at_high,
            srsi_at_low=srsi_at_low,
            srsi_at_high=srsi_at_high,
            prev_low_price=prev_low_price,
            prev_high_price=prev_high_price,
            prev_low_rsi=prev_low_rsi,
            prev_high_rsi=prev_high_rsi,
            prev_low_srsi=prev_low_srsi,
            prev_high_srsi=prev_high_srsi,
            dist_ok_low=dist_ok_low,
            dist_ok_high=dist_ok_high,
            prev_low_bar=prev_low_bar,
            prev_high_bar=prev_high_bar,
            bar_positions=bar_positions,
            price_low_arr=price_low_arr,
            price_high_arr=price_high_arr,
            rsi_arr=rsi_arr,
            p_len=p_len,
            config=config,
        )

        result[f"{scale}_bull_regular"] = sigs["bull_regular"]
        result[f"{scale}_bear_regular"] = sigs["bear_regular"]
        result[f"{scale}_bull_hidden"] = sigs["bull_hidden"]
        result[f"{scale}_bear_hidden"] = sigs["bear_hidden"]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Micro and Macro RSI Divergence signals")
    p.add_argument("--csv", required=True, help="Input CSV path (date,open,high,low,close[,volume])")
    p.add_argument("--output", default="-", help="Output CSV path (default: stdout)")
    p.add_argument("--rsi-length", type=int, default=14)
    p.add_argument("--smooth-type", default="EMA", choices=["SMA", "EMA", "RMA", "WMA", "HMA", "None"])
    p.add_argument("--smooth-length", type=int, default=60)
    p.add_argument("--pivot-source", default="hl", choices=["hl", "close"])
    p.add_argument("--p-len", type=int, default=2)
    p.add_argument("--p-len-macro", type=int, default=10)
    p.add_argument("--min-dist", type=int, default=5)
    p.add_argument("--max-dist", type=int, default=100)
    p.add_argument("--min-rsi-diff", type=float, default=4.0)
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    df = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    cfg = Config(
        rsi_length=args.rsi_length,
        smooth_type=args.smooth_type,
        smooth_length=args.smooth_length,
        pivot_source=args.pivot_source,
        p_len=args.p_len,
        p_len_macro=args.p_len_macro,
        min_dist=args.min_dist,
        max_dist=args.max_dist,
        min_rsi_diff=args.min_rsi_diff,
    )
    out = compute(df, cfg)
    if args.output == "-":
        out.to_csv(sys.stdout)
    else:
        out.to_csv(args.output)
