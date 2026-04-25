"""Intraday session helpers for sub-daily (15m, 5m, 1h, …) bar frames.

All functions here assume a ``DatetimeIndex`` produced by an intraday
``MarketDataPort.fetch_ohlcv`` — ideally already tz-aware in
``America/New_York``, but ``ensure_ny_tz`` will normalize a tz-naive or
UTC frame on the way in. The rest of the module is intentionally
session-date centric (bars grouped by their NY calendar date) so that
"previous day" semantics don't depend on arbitrary 24-hour windows.

Exports:
- ``ensure_ny_tz(df)`` — idempotent tz conversion
- ``regular_session_only(df)`` — filter to 09:30–15:59 ET
- ``session_dates(df)`` — np.array of NY calendar dates aligned to the index
- ``prev_session_ohlc(df, at_ts)`` — previous session's O/H/L/C or ``None``
- ``session_ohlc_so_far(df, at_ts)`` — today's O/H/L accumulated up to ``at_ts``
- ``resample_to_daily(df)`` — daily OHLCV aggregation of an intraday frame
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"
SESSION_OPEN = "09:30"
SESSION_CLOSE = "15:59"  # last 15m bar starts at 15:45; keep the whole hour


@dataclass(frozen=True)
class SessionOHLC:
    open: float
    high: float
    low: float
    close: float


def ensure_ny_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Return a view of ``df`` whose index is tz-aware in America/New_York.

    tz-naive inputs are *localized* (assumed to be NY wall time — this is
    what yfinance returns for intraday after our adapter's normalization).
    Other-timezone inputs are *converted*. No-op if already in NY.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"expected DatetimeIndex, got {type(df.index).__name__}"
        )
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ)
    elif str(idx.tz) != NY_TZ:
        idx = idx.tz_convert(NY_TZ)
    else:
        return df
    out = df.copy()
    out.index = idx
    return out


def regular_session_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular US equity session bars (09:30–16:00 ET).

    Safe to call on a frame that's already filtered; keeps the original
    tz. Assumes ``ensure_ny_tz`` has been applied first (or that the
    frame is already in NY time).
    """
    return df.between_time(SESSION_OPEN, SESSION_CLOSE)


def session_dates(df: pd.DataFrame) -> np.ndarray:
    """Return an array of NY calendar dates, one per row of ``df``.

    The array is aligned with ``df.index`` — callers use it to group or
    to find the first/last bar of a given session.
    """
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ)
    elif str(idx.tz) != NY_TZ:
        idx = idx.tz_convert(NY_TZ)
    return np.array([ts.date() for ts in idx])


def prev_session_ohlc(
    df: pd.DataFrame, at_ts: pd.Timestamp
) -> SessionOHLC | None:
    """OHLC of the session preceding ``at_ts``'s session date.

    Returns ``None`` if ``at_ts`` is on the first session in ``df``
    (no prior history). ``at_ts`` does not need to match a bar exactly —
    its NY date is used to identify "today's" session.
    """
    at_date = _to_ny_date(at_ts)
    dates = session_dates(df)
    prior = dates < at_date
    if not prior.any():
        return None
    last_prior_date = dates[prior].max()
    mask = dates == last_prior_date
    window = df.loc[mask]
    if window.empty:
        return None
    return SessionOHLC(
        open=float(window["Open"].iloc[0]),
        high=float(window["High"].max()),
        low=float(window["Low"].min()),
        close=float(window["Close"].iloc[-1]),
    )


def session_ohlc_so_far(
    df: pd.DataFrame, at_ts: pd.Timestamp
) -> SessionOHLC | None:
    """Open/high/low/close of today's session up to **and including** ``at_ts``.

    "Today" is the NY calendar date of ``at_ts``. Returns ``None`` if no
    bar of today's session is present at or before ``at_ts``.
    """
    at_date = _to_ny_date(at_ts)
    at_ts_ny = _to_ny_ts(at_ts)
    dates = session_dates(df)
    mask = (dates == at_date) & (df.index <= at_ts_ny)
    window = df.loc[mask]
    if window.empty:
        return None
    return SessionOHLC(
        open=float(window["Open"].iloc[0]),
        high=float(window["High"].max()),
        low=float(window["Low"].min()),
        close=float(window["Close"].iloc[-1]),
    )


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an intraday frame to daily OHLCV.

    Grouping is by NY calendar date (not by a rolling 24h window) so
    each row corresponds to one trading session. The resulting index is
    a tz-naive ``DatetimeIndex`` of session dates — matching the
    convention of the daily ``YFinanceAdapter`` path, so daily-built
    indicators can consume it unchanged.
    """
    dates = session_dates(df)
    grouped = df.groupby(dates)
    daily = pd.DataFrame(
        {
            "Open": grouped["Open"].first(),
            "High": grouped["High"].max(),
            "Low": grouped["Low"].min(),
            "Close": grouped["Close"].last(),
            "Volume": grouped["Volume"].sum(),
        }
    )
    daily.index = pd.to_datetime(daily.index)
    return daily


def _to_ny_date(ts: pd.Timestamp) -> date:
    ts = _to_ny_ts(ts)
    return ts.date()


def _to_ny_ts(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        return ts.tz_localize(NY_TZ)
    if str(ts.tz) != NY_TZ:
        return ts.tz_convert(NY_TZ)
    return ts
