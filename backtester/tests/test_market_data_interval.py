"""Interval propagation for the market-data layer.

Covers three guarantees needed by the 15m strategy work:

1. ``MarketDataPort.fetch_ohlcv`` accepts ``interval`` as a kwarg and
   ``CachedMarketDataAdapter`` forwards it to the upstream.
2. Different intervals write to different cache files so 1d and 15m
   can't stomp each other on disk.
3. The legacy ``"1d"`` cache path is byte-for-byte identical to the
   pre-interval scheme, so parquet files created before this change
   keep hitting the cache.
"""

from __future__ import annotations

import inspect
from datetime import date

import pandas as pd
import pytest

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.domain.ports import MarketDataPort


class _StubUpstream(MarketDataPort):
    """Records every fetch_ohlcv call; returns a tiny OHLCV frame."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date, str]] = []

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        self.calls.append((symbol, start, end, interval))
        idx = pd.date_range(start=start, periods=2, freq="D")
        return pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.5, 2.5],
                "Low": [0.9, 1.9],
                "Close": [1.2, 2.2],
                "Volume": [100, 200],
            },
            index=idx,
        )


def test_port_signature_has_interval_kwarg() -> None:
    sig = inspect.signature(MarketDataPort.fetch_ohlcv)
    assert "interval" in sig.parameters
    assert sig.parameters["interval"].default == "1d"


def test_cache_forwards_interval_to_upstream(tmp_path) -> None:
    upstream = _StubUpstream()
    cache = CachedMarketDataAdapter(upstream, cache_dir=tmp_path)

    start, end = date(2023, 1, 1), date(2023, 1, 31)
    cache.fetch_ohlcv("AAPL", start, end, interval="15m")

    assert upstream.calls == [("AAPL", start, end, "15m")]


def test_distinct_intervals_write_distinct_cache_files(tmp_path) -> None:
    upstream = _StubUpstream()
    cache = CachedMarketDataAdapter(upstream, cache_dir=tmp_path)

    start, end = date(2023, 1, 1), date(2023, 1, 31)
    cache.fetch_ohlcv("AAPL", start, end, interval="1d")
    cache.fetch_ohlcv("AAPL", start, end, interval="15m")

    parquet_files = sorted(p.name for p in tmp_path.glob("AAPL_*.parquet"))
    assert len(parquet_files) == 2, parquet_files

    # Both hit the upstream exactly once — second 15m call is cold.
    assert len(upstream.calls) == 2

    # Re-issuing the same 15m request must hit the cache, not upstream.
    cache.fetch_ohlcv("AAPL", start, end, interval="15m")
    assert len(upstream.calls) == 2


def test_legacy_1d_cache_path_unchanged(tmp_path) -> None:
    """The pre-interval key was ``{symbol}_{start}_{end}``. We keep that
    exact form for ``interval="1d"`` so previously written parquet
    caches keep resolving after this change."""
    upstream = _StubUpstream()
    cache = CachedMarketDataAdapter(upstream, cache_dir=tmp_path)

    start, end = date(2023, 1, 1), date(2023, 1, 31)
    expected = cache._cache_path("AAPL", start, end, interval="1d")
    legacy = cache._cache_path("AAPL", start, end)  # default arg

    assert expected == legacy


@pytest.mark.parametrize("interval", ["1h", "30m", "15m", "5m"])
def test_intraday_cache_key_includes_interval(tmp_path, interval: str) -> None:
    upstream = _StubUpstream()
    cache = CachedMarketDataAdapter(upstream, cache_dir=tmp_path)

    start, end = date(2023, 1, 1), date(2023, 1, 31)
    daily = cache._cache_path("AAPL", start, end, interval="1d")
    intraday = cache._cache_path("AAPL", start, end, interval=interval)

    assert daily != intraday
