import numpy as np
import pandas as pd
import pytest

from config.rsi_div_config import Config
from indicators.rsi_divergence import SIGNAL_COLUMNS, compute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_cfg(**kwargs) -> Config:
    defaults = dict(
        rsi_length=14,
        smooth_type="EMA",
        smooth_length=20,  # shorter smooth for test speed
        p_len=2,
        p_len_macro=5,     # smaller macro for test data
        min_dist=3,
        max_dist=80,
        min_rsi_diff=0.0,  # no diff filter unless explicitly set
    )
    defaults.update(kwargs)
    return Config(**defaults)


# ---------------------------------------------------------------------------
# Column presence and types
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_signal_columns_present(self, synthetic_df):
        result = compute(synthetic_df, _default_cfg())
        for col in SIGNAL_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_signal_columns_are_bool(self, synthetic_df):
        result = compute(synthetic_df, _default_cfg())
        for col in SIGNAL_COLUMNS:
            assert result[col].dtype == bool or result[col].dtype == np.bool_, col

    def test_ohlcv_unchanged(self, synthetic_df):
        result = compute(synthetic_df, _default_cfg())
        for col in ["open", "high", "low", "close"]:
            pd.testing.assert_series_equal(result[col], synthetic_df[col])

    def test_index_preserved(self, synthetic_df):
        result = compute(synthetic_df, _default_cfg())
        pd.testing.assert_index_equal(result.index, synthetic_df.index)


# ---------------------------------------------------------------------------
# No signals on flat data
# ---------------------------------------------------------------------------

class TestFlatData:
    def test_no_signals_flat(self, flat_df):
        result = compute(flat_df, _default_cfg())
        for col in SIGNAL_COLUMNS:
            assert not result[col].any(), f"Unexpected signal in {col} on flat data"


# ---------------------------------------------------------------------------
# At least some signals fire on oscillating data
# ---------------------------------------------------------------------------

class TestSignalsFire:
    def test_micro_signals_fire(self, oscillating_df):
        cfg = _default_cfg(min_rsi_diff=0.0)
        result = compute(oscillating_df, cfg)
        micro_cols = [c for c in SIGNAL_COLUMNS if c.startswith("micro")]
        total = sum(result[c].sum() for c in micro_cols)
        assert total > 0, "No micro signals fired — check oscillating_df fixture"

    def test_macro_signals_fire(self, oscillating_df):
        cfg = _default_cfg(min_rsi_diff=0.0)
        result = compute(oscillating_df, cfg)
        macro_cols = [c for c in SIGNAL_COLUMNS if c.startswith("macro")]
        total = sum(result[c].sum() for c in macro_cols)
        assert total > 0, "No macro signals fired — check oscillating_df fixture"

    def test_signals_fire_on_random_walk(self, synthetic_df):
        """Random walk over 500 bars reliably produces some divergences."""
        cfg = _default_cfg(min_rsi_diff=0.0)
        result = compute(synthetic_df, cfg)
        total = sum(result[c].sum() for c in SIGNAL_COLUMNS)
        assert total > 0, "No signals fired on 500-bar random walk"


# ---------------------------------------------------------------------------
# Macro produces <= signals compared to micro (smaller p_len = more sensitive)
# ---------------------------------------------------------------------------

class TestMacroSubset:
    def test_macro_le_micro_count(self, oscillating_df):
        cfg = _default_cfg(min_rsi_diff=0.0)
        result = compute(oscillating_df, cfg)
        micro_total = sum(result[c].sum() for c in SIGNAL_COLUMNS if "micro" in c)
        macro_total = sum(result[c].sum() for c in SIGNAL_COLUMNS if "macro" in c)
        assert macro_total <= micro_total


# ---------------------------------------------------------------------------
# Distance filter
# ---------------------------------------------------------------------------

class TestDistanceFilter:
    def test_large_min_dist_suppresses(self, oscillating_df):
        cfg_open = _default_cfg(min_dist=3, max_dist=500, min_rsi_diff=0.0)
        cfg_tight = _default_cfg(min_dist=200, max_dist=500, min_rsi_diff=0.0)
        res_open = compute(oscillating_df, cfg_open)
        res_tight = compute(oscillating_df, cfg_tight)
        for col in SIGNAL_COLUMNS:
            assert res_tight[col].sum() <= res_open[col].sum()


# ---------------------------------------------------------------------------
# RSI-diff filter
# ---------------------------------------------------------------------------

class TestRsiDiffFilter:
    def test_high_diff_reduces_signals(self, oscillating_df):
        cfg_low = _default_cfg(min_rsi_diff=0.0)
        cfg_high = _default_cfg(min_rsi_diff=30.0)
        res_low = compute(oscillating_df, cfg_low)
        res_high = compute(oscillating_df, cfg_high)
        for col in SIGNAL_COLUMNS:
            assert res_high[col].sum() <= res_low[col].sum()


# ---------------------------------------------------------------------------
# Config validation errors
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_invalid_p_len(self):
        with pytest.raises(ValueError):
            Config(p_len=0)

    def test_invalid_dist(self):
        with pytest.raises(ValueError):
            Config(min_dist=50, max_dist=50)

    def test_invalid_smooth_type(self):
        with pytest.raises(ValueError):
            Config(smooth_type="INVALID")

    def test_invalid_rsi_source(self):
        with pytest.raises(ValueError):
            Config(rsi_source="INVALID")


# ---------------------------------------------------------------------------
# Insufficient data error
# ---------------------------------------------------------------------------

class TestInsufficientData:
    def test_too_few_bars_raises(self):
        tiny = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]},
            index=pd.date_range("2020-01-01", periods=1, freq="D"),
        )
        with pytest.raises(ValueError, match="bars"):
            compute(tiny, _default_cfg())


# ---------------------------------------------------------------------------
# Non-monotonic index raises
# ---------------------------------------------------------------------------

class TestBadIndex:
    def test_non_monotonic_raises(self, synthetic_df):
        df = synthetic_df.iloc[::-1].copy()
        with pytest.raises(ValueError, match="monotonically"):
            compute(df, _default_cfg())


# ---------------------------------------------------------------------------
# Optional filters reduce signal count
# ---------------------------------------------------------------------------

class TestOptionalFilters:
    def test_rsi_50_filter_reduces(self, oscillating_df):
        cfg_off = _default_cfg(filter_rsi_50=False, min_rsi_diff=0.0)
        cfg_on = _default_cfg(filter_rsi_50=True, min_rsi_diff=0.0)
        res_off = compute(oscillating_df, cfg_off)
        res_on = compute(oscillating_df, cfg_on)
        for col in SIGNAL_COLUMNS:
            assert res_on[col].sum() <= res_off[col].sum()
