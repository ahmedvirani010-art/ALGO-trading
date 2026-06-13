import numpy as np
import pandas as pd
import pytest

from indicators.ta_utils import (
    compute_rsi,
    ema,
    hma,
    pivot_high,
    pivot_low,
    rma,
    sma,
    valuewhen,
    wma,
)


def _series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# sma
# ---------------------------------------------------------------------------

class TestSma:
    def test_basic(self):
        s = _series([1, 2, 3, 4, 5])
        result = sma(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_nan_prefix_length(self):
        s = _series(range(20))
        result = sma(s, 5)
        assert result.iloc[:4].isna().all()
        assert result.iloc[4:].notna().all()


# ---------------------------------------------------------------------------
# ema
# ---------------------------------------------------------------------------

class TestEma:
    def test_flat_series(self):
        s = _series([5.0] * 50)
        result = ema(s, 10)
        # EMA of constant = constant
        assert result.iloc[-1] == pytest.approx(5.0, rel=1e-6)

    def test_increasing(self):
        s = _series(range(1, 101))
        result = ema(s, 10)
        # EMA should be below last value in rising series
        assert result.iloc[-1] < 100


# ---------------------------------------------------------------------------
# rma
# ---------------------------------------------------------------------------

class TestRma:
    def test_flat_series(self):
        s = _series([10.0] * 100)
        result = rma(s, 14)
        assert result.iloc[-1] == pytest.approx(10.0, rel=1e-6)

    def test_alpha(self):
        """RMA alpha = 1/length; verify via manual recursion."""
        s = _series([1.0, 2.0, 3.0, 4.0, 5.0])
        length = 3
        alpha = 1.0 / length
        result = rma(s, length)
        # Manual: ewm(alpha=alpha, adjust=False)
        manual = s.ewm(alpha=alpha, adjust=False).mean()
        pd.testing.assert_series_equal(result, manual)


# ---------------------------------------------------------------------------
# wma
# ---------------------------------------------------------------------------

class TestWma:
    def test_hand_verify(self):
        # WMA([1,2,3,4], length=4): weights=[1,2,3,4]/10 → dot=[1,4,9,16]/10 = 3.0
        s = _series([1.0, 2.0, 3.0, 4.0])
        result = wma(s, 4)
        assert result.iloc[-1] == pytest.approx(3.0)

    def test_nan_prefix(self):
        s = _series(range(10))
        result = wma(s, 4)
        assert result.iloc[:3].isna().all()
        assert result.iloc[3:].notna().all()

    def test_flat(self):
        s = _series([7.0] * 20)
        result = wma(s, 5)
        assert result.iloc[-1] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# hma
# ---------------------------------------------------------------------------

class TestHma:
    def test_flat_series(self):
        s = _series([42.0] * 100)
        result = hma(s, 16)
        # HMA of constant should equal constant after warmup
        valid = result.dropna()
        assert (valid - 42.0).abs().max() < 1e-6

    def test_nan_propagation(self):
        s = _series(range(50))
        result = hma(s, 9)
        assert result.iloc[0] is np.nan or pd.isna(result.iloc[0])


# ---------------------------------------------------------------------------
# compute_rsi
# ---------------------------------------------------------------------------

class TestComputeRsi:
    def test_range(self, synthetic_df):
        from indicators.ta_utils import compute_rsi
        rsi = compute_rsi(synthetic_df["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_monotone_up(self):
        close = _series(np.linspace(1, 100, 200))
        rsi = compute_rsi(close, 14)
        # strictly rising price → RSI should approach 100
        assert rsi.dropna().iloc[-1] > 95

    def test_monotone_down(self):
        close = _series(np.linspace(100, 1, 200))
        rsi = compute_rsi(close, 14)
        assert rsi.dropna().iloc[-1] < 5

    def test_nan_prefix(self):
        close = _series(np.ones(50))
        rsi = compute_rsi(close, 14)
        # First bar: diff is NaN so gain/loss are NaN → rsi NaN
        assert pd.isna(rsi.iloc[0])


# ---------------------------------------------------------------------------
# pivot_low / pivot_high
# ---------------------------------------------------------------------------

class TestPivotLow:
    def test_single_pivot(self):
        # minimum at position 5; with L=R=2, confirmed at position 7
        vals = [10, 9, 8, 7, 6, 2, 6, 7, 8, 9, 10]
        s = _series(vals)
        result = pivot_low(s, 2, 2)
        # pivot confirmation bar is index 7 (0-based), value = 2.0
        assert result.iloc[7] == pytest.approx(2.0)
        # no pivot at other bars (except if coincidentally a pivot too)
        assert pd.isna(result.iloc[6])
        assert pd.isna(result.iloc[8])

    def test_nan_prefix(self):
        s = _series(range(20))
        result = pivot_low(s, 3, 3)
        assert result.iloc[:5].isna().all()  # first L+R = 6 bars but rolling(7) needs 7

    def test_no_false_pivot(self):
        # strictly increasing — no pivot low possible
        s = _series(np.arange(20, dtype=float))
        result = pivot_low(s, 2, 2)
        assert result.isna().all()


class TestPivotHigh:
    def test_single_pivot(self):
        vals = [2, 3, 4, 5, 10, 5, 4, 3, 2, 1, 2]
        s = _series(vals)
        result = pivot_high(s, 2, 2)
        # maximum at index 4, confirmed at index 6
        assert result.iloc[6] == pytest.approx(10.0)

    def test_no_false_pivot(self):
        s = _series(np.arange(20, dtype=float))
        result = pivot_high(s, 2, 2)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# valuewhen
# ---------------------------------------------------------------------------

class TestValuewhen:
    def _make(self, cond_list, val_list):
        idx = pd.date_range("2020-01-01", periods=len(cond_list), freq="D")
        cond = pd.Series(cond_list, index=idx, dtype=bool)
        val = pd.Series(val_list, index=idx, dtype=float)
        return cond, val

    def test_occurrence_0(self):
        cond, val = self._make(
            [False, True, False, True, False],
            [0, 10, 20, 30, 40],
        )
        result = valuewhen(cond, val, occurrence=0)
        # at bar 1 (first True): returns val[1]=10
        assert result.iloc[1] == pytest.approx(10.0)
        # at bar 3 (second True): returns val[3]=30
        assert result.iloc[3] == pytest.approx(30.0)
        # bar 4 (after second True, forward-filled): returns 30
        assert result.iloc[4] == pytest.approx(30.0)
        # before any True: NaN
        assert pd.isna(result.iloc[0])

    def test_occurrence_1(self):
        cond, val = self._make(
            [False, True, False, True, False, True],
            [0, 10, 20, 30, 40, 50],
        )
        result = valuewhen(cond, val, occurrence=1)
        # at bar 3 (second True): returns val at first True = 10
        assert result.iloc[3] == pytest.approx(10.0)
        # at bar 5 (third True): returns val at second True = 30
        assert result.iloc[5] == pytest.approx(30.0)
        # bar 4 (between second and third): forward-filled from bar 3 → 10
        assert result.iloc[4] == pytest.approx(10.0)
        # bars 0,1,2: fewer than 2 True bars → NaN
        assert result.iloc[:3].isna().all()

    def test_all_false(self):
        cond, val = self._make([False] * 10, list(range(10)))
        result = valuewhen(cond, val, occurrence=0)
        assert result.isna().all()

    def test_single_true_occurrence_1(self):
        cond, val = self._make(
            [False, True, False, False],
            [0, 99, 0, 0],
        )
        result = valuewhen(cond, val, occurrence=1)
        # only 1 True bar → can never satisfy occurrence=1 → all NaN
        assert result.isna().all()
