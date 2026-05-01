"""Trading-session metadata for equity markets.

A :class:`MarketCalendar` is a small value object carrying the three
pieces of information that every detector / strategy / chart in this
codebase needs to interpret an OHLC frame in its local context:

- IANA timezone (so tz-naive index lookups stay unambiguous)
- Regular-trading-hours window (open / close in *local* time)
- Display name

Three pre-built instances are exported:

    NY   — US cash-equity session (NYSE / Nasdaq).  09:30–16:00 ET
    KR   — Korean continuous session (KOSPI/KOSDAQ).  09:00–15:30 KST

and a suffix-based dispatcher :func:`market_for_ticker` that maps
yfinance ticker conventions (``005930.KS`` → KR, ``AAPL`` → NY) onto
the right calendar.

Adding a new market is a one-line change here — define another
``MarketCalendar(...)``  instance and (if it has its own ticker
suffix) extend ``market_for_ticker``. Detector / strategy / chart
code already accepts any calendar instance, so no downstream edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketCalendar:
    """Where a ticker trades, and during which hours.

    Frozen + hashable so callers can stash it in caches /
    dispatch tables.

    Attributes
    ----------
    name:
        Human-readable label (``"NY"``, ``"KR"``). Used in chart
        titles, log lines, and debugging — never as a lookup key.
    tz:
        IANA timezone identifier. ``ZoneInfo(tz)`` is exposed via
        :attr:`zoneinfo` for callers that need an actual ``tzinfo``.
    rth_open / rth_close:
        Regular trading hours in **local** wall-clock time. The RTH
        window is half-open ``[rth_open, rth_close)`` to match the
        existing detector convention (a 15m bar starting exactly at
        ``rth_close`` is post-RTH).
    """

    name: str
    tz: str
    rth_open: time
    rth_close: time

    @property
    def zoneinfo(self) -> ZoneInfo:
        return ZoneInfo(self.tz)


# US cash equity session — NYSE / Nasdaq. Continuous, no lunch break.
NY = MarketCalendar(
    name="NY",
    tz="America/New_York",
    rth_open=time(9, 30),
    rth_close=time(16, 0),
)


# Korea — KOSPI / KOSDAQ. The official session is 09:00–15:30 KST
# (continuous 09:00–15:20 + closing auction 15:20–15:30), but our
# data path through EODHD only carries clean continuous-bar data up
# to the 14:45 bar (covering 14:45–15:00). The 15:00–15:30 window is
# represented in EODHD only as a single placeholder bar at 15:00
# with ``V=None`` and OHLC all equal to the auction close — that's
# already filtered by ``EODHDAdapter`` and would be empty anyway.
# Setting ``rth_close = 15:00`` here keeps the RTH gate, session-
# boundary detection, and ``force_close_at_session_end`` aligned
# with what bars actually exist; revisit if/when we add a richer
# KR data source (KIS / KRX direct).
KR = MarketCalendar(
    name="KR",
    tz="Asia/Seoul",
    rth_open=time(9, 0),
    rth_close=time(15, 0),
)


def market_for_ticker(ticker: str) -> MarketCalendar:
    """Resolve the market from a yfinance ticker suffix.

    yfinance convention: ``.KS`` for KOSPI, ``.KQ`` for KOSDAQ.
    Anything else (no suffix or any other suffix) defaults to NY —
    the vast majority of US-style tickers are bare symbols with no
    suffix.
    """
    upper = ticker.upper()
    if upper.endswith(".KS") or upper.endswith(".KQ"):
        return KR
    return NY
