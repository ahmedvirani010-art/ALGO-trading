"""
TA primitive helpers translated from Pine Script v6 semantics.

All functions accept pd.Series indexed identically to the input DataFrame and
return pd.Series with NaN padding at the front matching Pine Script warmup.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def rma(src: pd.Series, length: int) -> pd.Series:
    """Wilder's Moving Average (alpha = 1/length). Seeds from first observation."""
    return src.ewm(alpha=1.0 / length, adjust=False).mean()


def sma(src: pd.Series, length: int) -> pd.Series:
    return src.rolling(length, min_periods=length).mean()


def ema(src: pd.Series, length: int) -> pd.Series:
    return src.ewm(span=length, adjust=False).mean()


def wma(src: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1, dtype=float)
    weights /= weights.sum()
    return src.rolling(length, min_periods=length).apply(
        lambda x: np.dot(x, weights), raw=True
    )


def hma(src: pd.Series, length: int) -> pd.Series:
    half = max(1, length // 2)
    sqrt_len = max(1, int(np.floor(np.sqrt(length))))
    return wma(2.0 * wma(src, half) - wma(src, length), sqrt_len)


def smooth_rsi(rsi: pd.Series, smooth_type: str, length: int) -> pd.Series:
    dispatch = {
        "SMA": sma,
        "EMA": ema,
        "RMA": rma,
        "WMA": wma,
        "HMA": hma,
    }
    if smooth_type == "None":
        return rsi.copy()
    fn = dispatch.get(smooth_type)
    if fn is None:
        raise ValueError(f"Unknown smooth_type: {smooth_type}")
    return fn(rsi, length)


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def compute_source(df: pd.DataFrame, source: str) -> pd.Series:
    if source == "close":
        return df["close"]
    if source == "open":
        return df["open"]
    if source == "high":
        return df["high"]
    if source == "low":
        return df["low"]
    if source == "hl2":
        return (df["high"] + df["low"]) / 2.0
    if source == "hlc3":
        return (df["high"] + df["low"] + df["close"]) / 3.0
    if source == "ohlc4":
        return (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    raise ValueError(f"Unknown source: {source}")


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

def compute_rsi(src: pd.Series, length: int) -> pd.Series:
    delta = src.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss
    rsi_vals = np.where(avg_loss == 0, 100.0, np.where(avg_gain == 0, 0.0, 100.0 - 100.0 / (1.0 + rs)))
    return pd.Series(rsi_vals, index=src.index, dtype=float)


# ---------------------------------------------------------------------------
# Pivot detection  (exact Pine Script ta.pivotlow / ta.pivothigh semantics)
#
# ta.pivotlow(src, L, R) at bar i returns src[i-R] if that bar is the
# minimum of src[i-L-R .. i]; else na.  We emit the result at the
# *confirmation* bar i (R bars after the actual pivot bar).
# ---------------------------------------------------------------------------

def pivot_low(src: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    W = left_bars + right_bars + 1
    rolling_min = src.rolling(W, min_periods=W).min()
    center_val = src.shift(right_bars)
    is_pivot = (center_val == rolling_min) & rolling_min.notna()
    vals = np.where(is_pivot, center_val, np.nan)
    return pd.Series(vals, index=src.index, dtype=float)


def pivot_high(src: pd.Series, left_bars: int, right_bars: int) -> pd.Series:
    W = left_bars + right_bars + 1
    rolling_max = src.rolling(W, min_periods=W).max()
    center_val = src.shift(right_bars)
    is_pivot = (center_val == rolling_max) & rolling_max.notna()
    vals = np.where(is_pivot, center_val, np.nan)
    return pd.Series(vals, index=src.index, dtype=float)


# ---------------------------------------------------------------------------
# valuewhen  (Pine Script ta.valuewhen semantics)
#
# occurrence=0 → most recent True bar (can be current)
# occurrence=1 → second most recent True bar (used for "previous pivot")
# ---------------------------------------------------------------------------

def valuewhen(cond: pd.Series, val: pd.Series, occurrence: int = 1) -> pd.Series:
    """
    At each bar, return val at the occurrence-th most recent bar where cond
    is True (0 = most recent, 1 = one before that).
    """
    true_idx = cond[cond].index
    result = pd.Series(np.nan, index=cond.index, dtype=float)
    if len(true_idx) <= occurrence:
        return result
    for k in range(occurrence, len(true_idx)):
        result.loc[true_idx[k]] = val.loc[true_idx[k - occurrence]]
    return result.ffill()


# ---------------------------------------------------------------------------
# Volatility & trend-strength indicators
# ---------------------------------------------------------------------------

def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (RMA)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return rma(tr, length)


def compute_adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """
    Average Directional Index.
    Returns DataFrame with columns: adx, plus_di, minus_di  (all 0-100 scale).
    ADX > 25 indicates a trending market; > 40 a strong trend.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_high = high.shift(1)
    prev_low  = low.shift(1)

    move_up   = high - prev_high
    move_down = prev_low - low

    plus_dm  = np.where((move_up > move_down) & (move_up > 0),  move_up,  0.0)
    minus_dm = np.where((move_down > move_up) & (move_down > 0), move_down, 0.0)

    plus_dm_s  = pd.Series(plus_dm,  index=df.index, dtype=float)
    minus_dm_s = pd.Series(minus_dm, index=df.index, dtype=float)

    atr14 = compute_atr(df, length)

    plus_di  = 100.0 * rma(plus_dm_s,  length) / atr14
    minus_di = 100.0 * rma(minus_dm_s, length) / atr14

    di_sum  = plus_di + minus_di
    dx = np.where(di_sum == 0, 0.0, 100.0 * (plus_di - minus_di).abs() / di_sum)
    dx_s = pd.Series(dx, index=df.index, dtype=float)
    adx = rma(dx_s, length)

    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
                        index=df.index)


def compute_sma_distance(close: pd.Series, length: int = 200) -> pd.Series:
    """% distance of price from its SMA. Positive = above (uptrend)."""
    sma200 = sma(close, length)
    return (close - sma200) / sma200 * 100.0
