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


def _strict_local_extrema(
    vals: np.ndarray, left: int, right: int, is_high: bool
) -> np.ndarray:
    """Vectorized strict local extremum mask.

    Equivalent to the original per-index loop that required
    ``pivot == window.max() AND sum(window == pivot) == 1``: a strict
    maximum over ``[i-left, i+right]`` (no ties anywhere in the
    window). Implemented by comparing each shifted neighbor against
    the center; ties or missing sides disqualify the index. Runs in
    O(n × (left + right)) with numpy primitives instead of a Python
    loop, which is 10-100× faster on the 2,500-bar × 500-ticker
    scans this module sees.
    """
    n = vals.size
    mask = np.ones(n, dtype=bool)
    cmp = np.greater if is_high else np.less
    # A pivot must be strictly {>/<} each neighbor in the window.
    # The original's unique-max guard is equivalent to strict
    # inequality against every other window element.
    for k in range(1, left + 1):
        shifted = np.full(n, np.nan)
        shifted[k:] = vals[:-k]
        with np.errstate(invalid="ignore"):
            mask &= cmp(vals, shifted)
    for k in range(1, right + 1):
        shifted = np.full(n, np.nan)
        shifted[:-k] = vals[k:]
        with np.errstate(invalid="ignore"):
            mask &= cmp(vals, shifted)
    # Disqualify edges that can't satisfy the window.
    mask[:left] = False
    mask[n - right :] = False
    return mask


def find_swing_highs(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.Series:
    highs = df["High"].to_numpy()
    mask = _strict_local_extrema(highs, left, right, is_high=True)
    out = np.where(mask, highs, np.nan)
    return pd.Series(out, index=df.index, name="swing_high")


def find_swing_lows(df: pd.DataFrame, left: int = 2, right: int = 2) -> pd.Series:
    lows = df["Low"].to_numpy()
    mask = _strict_local_extrema(lows, left, right, is_high=False)
    out = np.where(mask, lows, np.nan)
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


def last_n_swing_highs(
    swing_highs: pd.Series,
    upto_idx: int,
    lookback: int,
    right: int,
    n: int = 2,
) -> list[tuple[int, float]]:
    """Most recent ``n`` confirmable swing highs before ``upto_idx``,
    ordered oldest → newest.
    """
    cutoff = upto_idx - right
    if cutoff < 0:
        return []
    start = max(0, cutoff - lookback + 1)
    window = swing_highs.iloc[start : cutoff + 1]
    non_na = window.dropna()
    if len(non_na) == 0:
        return []
    tail = non_na.iloc[-n:]
    result: list[tuple[int, float]] = []
    for ts, val in tail.items():
        result.append((int(swing_highs.index.get_loc(ts)), float(val)))
    return result


def last_swing_high(
    swing_highs: pd.Series,
    upto_idx: int,
    lookback: int,
    right: int,
) -> tuple[int, float] | None:
    """Most recent confirmable swing high before ``upto_idx``.

    Symmetric to ``recent_swing_high`` but picks the LAST pivot in time
    within the window instead of the highest. Used by the resistance-break
    exit because the "immediate overhead pivot" — the one price has to
    clear to prove near-term strength — is the most recent one, not some
    weeks-old global maximum.
    """
    cutoff = upto_idx - right
    if cutoff < 0:
        return None
    start = max(0, cutoff - lookback + 1)
    window = swing_highs.iloc[start : cutoff + 1]
    non_na = window.dropna()
    if len(non_na) == 0:
        return None
    last_ts = non_na.index[-1]
    pivot_idx = swing_highs.index.get_loc(last_ts)
    return int(pivot_idx), float(non_na.iloc[-1])


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
