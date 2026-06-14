from dataclasses import dataclass
from typing import Literal

_VALID_SMOOTH = {"SMA", "EMA", "RMA", "WMA", "HMA", "None"}
_VALID_SOURCES = {"close", "open", "high", "low", "hl2", "hlc3", "ohlc4"}
_VALID_PIVOT_SRC = {"hl", "close"}


@dataclass(frozen=True)
class Config:
    # RSI
    rsi_length: int = 14
    rsi_source: str = "close"

    # Smoothed RSI
    smooth_type: str = "EMA"
    smooth_length: int = 60

    # Pivot detection
    pivot_source: str = "hl"   # "hl" = High/Low, "close" = Close only
    p_len: int = 2             # micro pivot bars each side
    p_len_macro: int = 10      # macro pivot bars each side

    # Distance filter
    min_dist: int = 5
    max_dist: int = 100

    # RSI difference threshold
    min_rsi_diff: float = 4.0

    # Optional filters
    filter_rsi_50: bool = False
    filter_smooth_rsi_50: bool = False
    filter_rsi_direction: bool = False
    filter_smooth_rsi_direction: bool = False

    def __post_init__(self) -> None:
        if self.rsi_length < 1:
            raise ValueError("rsi_length must be >= 1")
        if self.rsi_source not in _VALID_SOURCES:
            raise ValueError(f"rsi_source must be one of {_VALID_SOURCES}")
        if self.smooth_type not in _VALID_SMOOTH:
            raise ValueError(f"smooth_type must be one of {_VALID_SMOOTH}")
        if self.smooth_length < 1:
            raise ValueError("smooth_length must be >= 1")
        if self.pivot_source not in _VALID_PIVOT_SRC:
            raise ValueError(f"pivot_source must be one of {_VALID_PIVOT_SRC}")
        if self.p_len < 1:
            raise ValueError("p_len must be >= 1")
        if self.p_len_macro < 1:
            raise ValueError("p_len_macro must be >= 1")
        if self.min_dist < 1:
            raise ValueError("min_dist must be >= 1")
        if self.max_dist <= self.min_dist:
            raise ValueError("max_dist must be > min_dist")
        if self.min_rsi_diff < 0:
            raise ValueError("min_rsi_diff must be >= 0")


def relaxed_config() -> "Config":
    """Looser thresholds that produce significantly more signals than the default."""
    return Config(
        rsi_length=14,
        smooth_type="EMA",
        smooth_length=20,   # was 60 — shorter smooth → RSI reacts faster
        p_len=1,            # was 2 — smaller micro pivot lookback
        p_len_macro=5,      # was 10 — smaller macro pivot lookback
        min_dist=3,         # was 5 — allow closer pivots
        max_dist=150,       # was 100 — allow more distant pivots
        min_rsi_diff=2.0,   # was 4.0 — accept weaker divergences
    )
