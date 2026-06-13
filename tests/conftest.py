import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(close: np.ndarray, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spread = np.abs(close * 0.002) + 0.01
    high = close + spread
    low = close - spread
    open_ = close + rng.uniform(-spread, spread)
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1e6},
        index=idx,
    )


@pytest.fixture
def flat_df():
    """500 bars of constant price — no divergences expected."""
    close = np.full(500, 100.0)
    return _make_ohlcv(close)


@pytest.fixture
def synthetic_df():
    """500-bar random walk for smoke tests."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(500) * 0.5)
    return _make_ohlcv(close)


@pytest.fixture
def oscillating_df():
    """
    500-bar series with a slow downtrend + oscillation that produces
    genuine RSI divergences: consecutive lows differ in price and in
    RSI (different descent rates → different momentum readings).
    """
    rng = np.random.default_rng(7)
    n = 500
    # Slow downtrend makes each swing low progressively lower in price.
    # Gradually lengthening oscillation period means descent is *less steep*
    # on later lows → RSI is higher even though price is lower → bull divergence.
    t = np.arange(n, dtype=float)
    trend = -0.04 * t
    # Increasing-period sine: cumulative phase so each cycle is 5% longer
    period = 25.0 + 0.06 * t
    phase = 2.0 * np.pi * np.cumsum(1.0 / period)
    oscillation = 8.0 * np.sin(phase)
    noise = rng.standard_normal(n) * 0.3
    close = np.clip(100.0 + trend + oscillation + noise, 0.5, None)
    return _make_ohlcv(close)


@pytest.fixture
def known_bull_df():
    """
    Hand-crafted 200-bar series with a deterministic regular bull divergence.

    Structure (bar indices):
      - Price makes a low at bar 20, then a LOWER low at bar 50.
      - RSI makes a low at bar 20, then a HIGHER low at bar 50
        (achieved by making the price drop steeper in the first leg,
        then gentler in the second).
    With p_len=2, the pivot at bar 20 is confirmed at bar 22,
    and the pivot at bar 50 is confirmed at bar 52.
    Distance between confirmation bars = 30, within [5, 100].
    """
    close = np.ones(200) * 100.0
    # first leg: sharp drop then recovery → low RSI
    close[10:21] = np.linspace(100, 80, 11)   # drop to 80
    close[21:35] = np.linspace(80, 100, 14)   # recover
    # second leg: gentle drop then recovery → higher RSI despite lower price
    close[35:51] = np.linspace(100, 75, 16)   # drop to 75 (lower price)
    close[51:65] = np.linspace(75, 100, 14)   # recover
    close[65:] = 100.0
    return _make_ohlcv(close)
