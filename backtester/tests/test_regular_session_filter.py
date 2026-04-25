"""``RegularSessionFilterAdapter`` filters sub-daily bars to RTH and
passes daily through unchanged.

The wrapper sits outside the cache so the parquet on disk keeps the
raw pre+post frame yfinance returned. These tests pin three
invariants that the rest of the data layer relies on:

1. ``interval="1d"`` is a pass-through — daily yfinance bars are
   already session-only by construction.
2. Sub-daily windows drop pre-market (04:00–09:29) and after-hours
   (16:00–20:00) bars. Bars at the session boundaries (09:30, 15:45)
   stay.
3. Upstream is called exactly once per fetch — the filter is purely
   a post-processing step, no extra round-trip.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from data.adapters.regular_session_filter import RegularSessionFilterAdapter
from data.domain.ports import MarketDataPort
from pattern.helpers.sessions import NY_TZ


class _RecordingUpstream(MarketDataPort):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.calls: list[tuple] = []

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        self.calls.append((symbol, start, end, interval))
        return self._df


def _intraday_with_pre_post() -> pd.DataFrame:
    """One-session 15m frame spanning 04:00 → 19:45 ET so the filter
    has pre-market, regular-session, and after-hours bars to choose
    between."""
    idx = pd.date_range(
        "2024-03-18 04:00", "2024-03-18 19:45", freq="15min", tz=NY_TZ
    )
    return pd.DataFrame(
        {
            "Open": range(len(idx)),
            "High": [x + 1 for x in range(len(idx))],
            "Low": [x - 1 for x in range(len(idx))],
            "Close": [x + 0.5 for x in range(len(idx))],
            "Volume": [1_000] * len(idx),
        },
        index=idx,
    )


def _daily_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-03-18", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": range(5),
            "High": [x + 1 for x in range(5)],
            "Low": [x - 1 for x in range(5)],
            "Close": [x + 0.5 for x in range(5)],
            "Volume": [1_000] * 5,
        },
        index=idx,
    )


def test_daily_passes_through_unchanged() -> None:
    upstream = _RecordingUpstream(_daily_frame())
    adapter = RegularSessionFilterAdapter(upstream)
    out = adapter.fetch_ohlcv(
        "AAPL", date(2024, 3, 18), date(2024, 3, 22), interval="1d"
    )
    assert len(out) == 5  # nothing dropped
    assert upstream.calls == [
        ("AAPL", date(2024, 3, 18), date(2024, 3, 22), "1d")
    ]


def test_intraday_drops_pre_and_post() -> None:
    upstream = _RecordingUpstream(_intraday_with_pre_post())
    adapter = RegularSessionFilterAdapter(upstream)
    out = adapter.fetch_ohlcv(
        "AAPL", date(2024, 3, 18), date(2024, 3, 18), interval="15m"
    )

    assert out.index.min().strftime("%H:%M") == "09:30"
    assert out.index.max().strftime("%H:%M") == "15:45"
    # 26 bars for a full regular session at 15m cadence
    # (09:30, 09:45, ..., 15:45)
    assert len(out) == 26


def test_filter_does_not_double_fetch() -> None:
    """Wrapper must be a pure post-processing step — one upstream
    call per public fetch, never two."""
    upstream = _RecordingUpstream(_intraday_with_pre_post())
    adapter = RegularSessionFilterAdapter(upstream)
    adapter.fetch_ohlcv(
        "AAPL", date(2024, 3, 18), date(2024, 3, 18), interval="15m"
    )
    adapter.fetch_ohlcv(
        "AAPL", date(2024, 3, 18), date(2024, 3, 18), interval="15m"
    )
    assert len(upstream.calls) == 2  # once per fetch, no extra reads


def test_interval_kwarg_forwarded_to_upstream() -> None:
    upstream = _RecordingUpstream(_intraday_with_pre_post())
    adapter = RegularSessionFilterAdapter(upstream)
    adapter.fetch_ohlcv(
        "AAPL", date(2024, 3, 18), date(2024, 3, 18), interval="5m"
    )
    assert upstream.calls[0][3] == "5m"
