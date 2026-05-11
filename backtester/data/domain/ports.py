from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class TickerFundamentals:
    """Per-ticker fundamentals snapshot used by stock-selection filters.

    Float / splits don't change intra-day so this is fetched once per
    ticker per scan and cached with a multi-day TTL upstream.

    ``float_shares`` is the *public float* (= shares outstanding minus
    insider/lock-up/treasury holdings) — the supply-side number Ross's
    Bull Flag setup keys off of (low float = more % move per dollar
    of buying pressure).

    ``splits`` matches ``yfinance.Ticker.splits``: a Series indexed by
    split-effective date, value = ratio (>1 = forward, <1 = reverse).
    Empty / None means no split history.
    """

    symbol: str
    float_shares: float | None
    splits: pd.Series | None


class FundamentalsPort(ABC):
    """Port: per-ticker fundamentals (float, splits) fetcher.

    Separate from :class:`MarketDataPort` because fundamentals come
    from different endpoints (EODHD ``/fundamentals``, Polygon
    ``/v3/reference/tickers``) and have a different cache profile —
    float updates rarely (weekly at best), so a multi-day TTL on the
    cache is appropriate.
    """

    @abstractmethod
    def fetch(self, symbol: str) -> TickerFundamentals:
        """Return a snapshot for ``symbol`` (e.g. ``"AAPL"``).

        On any data-source failure (HTTP 4xx/5xx, parse error, missing
        keys) the implementation returns ``TickerFundamentals(symbol,
        float_shares=None, splits=None)`` rather than raising — the
        strategy decides whether missing float disqualifies the ticker
        via ``require_float_filter``.
        """
        ...


class MarketDataPort(ABC):
    """Port: market data fetching interface."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        indexed by DatetimeIndex.

        ``interval`` follows yfinance conventions (e.g. ``"1d"``, ``"1h"``,
        ``"30m"``, ``"15m"``, ``"5m"``). For sub-daily intervals the
        returned DatetimeIndex is tz-aware in ``America/New_York`` and
        contains only regular US equity session bars (09:30–16:00 ET).
        Daily (``"1d"``) data keeps the legacy tz-naive contract.
        """
        ...


class UniverseProviderPort(ABC):
    """Port: ticker-universe provider.

    Resolves a universe identifier (e.g. ``"sp500"``, ``"nasdaq100"``) to
    the list of tickers belonging to that index. Used by multi-ticker
    strategies that scan an entire universe for opportunities.
    """

    @abstractmethod
    def get_tickers(self, universe: str) -> list[str]:
        """Return the list of tickers for the named universe."""
        ...
