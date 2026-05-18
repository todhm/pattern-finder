from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from data.domain.models import EarningsEvent, NewsEvent


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


@dataclass(frozen=True)
class RealtimeQuote:
    """Live snapshot of the *current session* OHLCV + last print.

    ``open / high / low`` reflect today-so-far, ``last`` is the most
    recent trade print (EODHD ``/real-time/`` calls this field
    ``close``, but during a live session it's the latest tick — we
    rename for clarity). ``volume`` is cumulative day volume.

    Used by Bull Flag live pages to display "current price" without
    paying for a 1m-bars round-trip, and to evaluate today's gap %
    when daily OHLCV hasn't settled yet.
    """

    symbol: str
    timestamp: datetime  # tz-aware UTC
    open: float
    high: float
    low: float
    last: float
    volume: int
    previous_close: float
    change: float
    change_pct: float

    @property
    def gap_pct(self) -> float:
        """(open − previousClose) / previousClose — today's gap-up %."""
        if self.previous_close <= 0:
            return 0.0
        return (self.open - self.previous_close) / self.previous_close


class RealtimeQuotePort(ABC):
    """Port: live current-session snapshot fetcher.

    Distinct from :class:`MarketDataPort` — that one returns a
    DataFrame of historical OHLCV bars. This one returns a single
    point-in-time quote per ticker (cheap, fast, fits a 5-30 ticker
    Bull Flag watchlist in one HTTP call when ``fetch_quotes`` is
    implemented in bulk).
    """

    @abstractmethod
    def fetch_quote(self, symbol: str) -> RealtimeQuote:
        """Return one ticker's live snapshot. Raises on data-source
        failure — callers wrap in try/except for per-ticker isolation."""
        ...

    @abstractmethod
    def fetch_quotes(self, symbols: list[str]) -> dict[str, RealtimeQuote]:
        """Bulk variant. Returns only successfully fetched symbols
        (missing keys = per-symbol failure). Implementations should
        batch when the upstream API supports multi-symbol calls."""
        ...


class EarningsCalendarPort(ABC):
    """Port: per-symbol earnings-announcement date fetcher.

    Used by strategies (notably Matt Diamond bull flag) to gate signals
    by proximity to earnings — Matt explicitly says earnings season is
    when catalyst-driven Bull Flags work best.

    Implementation contract: graceful degradation. If the upstream API
    is unreachable / unconfigured, return an empty list (not raise) so
    callers can choose to disable the gate rather than crash.
    """

    @abstractmethod
    def fetch_earnings(
        self,
        symbol: str,
        start: "date",
        end: "date",
    ) -> "list[EarningsEvent]":
        """Return all earnings reports for ``symbol`` whose
        ``report_date`` falls inside ``[start, end]``."""
        ...


class NewsCatalystPort(ABC):
    """Port: per-symbol news-event fetcher for catalyst gates.

    News data is sparse and rate-limited on most APIs, so the
    contract is windowed-and-paginated under the hood — callers just
    pass a date range. Returns ``[]`` on failure rather than raising
    (same graceful-degradation contract as EarningsCalendarPort).
    """

    @abstractmethod
    def fetch_news(
        self,
        symbol: str,
        start: "date",
        end: "date",
        limit_per_day: int | None = None,
    ) -> "list[NewsEvent]":
        """Return news items in ``[start, end]``.

        ``limit_per_day`` caps the items kept per calendar date so
        a noisy ticker doesn't blow out memory; ``None`` = keep all.
        Items are sorted by ``published_at`` ascending.
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
