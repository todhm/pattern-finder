"""Swing-pivot (fractal) detection and trendline fitting.

Bill Williams 5-bar fractal: a bar is a swing high if its High is strictly
greater than the High of the `left` bars before and the `right` bars after;
symmetric for swing low on Low. Detection has `right` bars of lag — a pivot
is only confirmable `right` bars after it prints. All helpers here are
causal: they never look at bars beyond the index the caller asks about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_swing_highs(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.Series:
    highs = df["High"].to_numpy()
    n = len(highs)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        pivot = highs[i]
        window = highs[i - left : i + right + 1]
        if pivot == window.max() and np.sum(window == pivot) == 1:
            out[i] = pivot
    return pd.Series(out, index=df.index, name="swing_high")


def find_swing_lows(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.Series:
    lows = df["Low"].to_numpy()
    n = len(lows)
    out = np.full(n, np.nan)
    for i in range(left, n - right):
        pivot = lows[i]
        window = lows[i - left : i + right + 1]
        if pivot == window.min() and np.sum(window == pivot) == 1:
            out[i] = pivot
    return pd.Series(out, index=df.index, name="swing_low")


def recent_swing_high(
    swing_highs: pd.Series,
    upto_idx: int,
    lookback: int,
    right: int,
) -> tuple[int, float] | None:
    """Highest swing high confirmable at `upto_idx`.

    Only pivots with index ≤ ``upto_idx - right`` are visible; swing highs
    printed more recently still need ``right`` future bars to confirm.
    Returns ``(pivot_idx, pivot_price)`` or ``None``.
    """
    cutoff = upto_idx - right
    if cutoff < 0:
        return None
    start = max(0, cutoff - lookback + 1)
    window = swing_highs.iloc[start : cutoff + 1]
    if window.notna().sum() == 0:
        return None
    rel = window.idxmax()
    pivot_idx = swing_highs.index.get_loc(rel)
    return int(pivot_idx), float(window.loc[rel])


def fit_lower_trendline(
    swing_lows: pd.Series,
    upto_idx: int,
    lookback: int,
    right: int,
    max_points: int = 3,
    min_points: int = 2,
) -> tuple[float, float, list[int]] | None:
    """Fit a line through the most recent confirmable swing lows.

    Returns ``(slope, intercept, pivot_positions)`` where
    ``y = slope * positional_idx + intercept``. Positional index is
    integer offset into the DataFrame, so extrapolation at any bar is
    just ``slope * i + intercept``. Returns ``None`` if fewer than
    ``min_points`` pivots are available in the window.
    """
    cutoff = upto_idx - right
    if cutoff < 0:
        return None
    start = max(0, cutoff - lookback + 1)
    window = swing_lows.iloc[start : cutoff + 1]
    mask = window.notna().to_numpy()
    if mask.sum() < min_points:
        return None
    positions = np.where(mask)[0] + start
    values = window.to_numpy()[mask]
    if len(positions) > max_points:
        positions = positions[-max_points:]
        values = values[-max_points:]
    if len(positions) == 2:
        x1, x2 = positions
        y1, y2 = values
        if x2 == x1:
            return None
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        slope, intercept = np.polyfit(positions.astype(float), values, 1)
    return float(slope), float(intercept), [int(p) for p in positions]
