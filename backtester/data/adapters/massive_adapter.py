"""Massive (formerly Polygon.io) market data adapter.

Polygon-compatible REST API providing US equity OHLCV with 10+ years
of history at 1m / 15m / etc. cadence — far beyond yfinance's 7-day
(1m) or 60-day (15m) caps. Used for sub-daily intervals on US
tickers; daily bars and Korean tickers stay on yfinance because the
depth there is fine, the cost is free, and Massive does not list
KRX/KOSDAQ.

Authentication
    ``MASSIVE_API_KEY`` env var, loaded into the backtester
    container via ``secrets/secret.env`` (see ``docker-compose.yaml``).

Endpoint
    ``GET {base}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{from}/{to}``

    Massive kept Polygon's URL/payload shape during the rebrand. The
    base URL defaults to ``https://api.polygon.io`` because that's
    what's still resolvable; override with ``MASSIVE_API_BASE`` if
    Massive ever moves the host.

Tz contract
    yfinance normalizes intraday frames to ``America/New_York``;
    we match that here so callers (the cache layer, RTH filter,
    detector) see a consistent index across sources. Polygon's
    ``t`` field is ms-epoch UTC.
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import httpx
import pandas as pd

from data.domain.ports import MarketDataPort

NY_TZ = "America/New_York"

# yfinance-style interval string → (multiplier, timespan) for the
# Polygon aggregates endpoint.
_INTERVAL_MAP: dict[str, tuple[int, str]] = {
    "1m": (1, "minute"),
    "2m": (2, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "60m": (60, "minute"),
    "1h": (1, "hour"),
    "1d": (1, "day"),
    "1wk": (1, "week"),
    "1mo": (1, "month"),
}


class MassiveAdapter(MarketDataPort):
    """Polygon/Massive REST client implementing :class:`MarketDataPort`."""

    DEFAULT_BASE_URL = "https://api.polygon.io"
    # ``limit`` is Polygon's per-request cap. 50,000 covers ~9 years
    # of 1m bars on a ticker so a single fetch usually suffices for
    # the windows this strategy uses; we paginate via ``next_url``
    # for anything bigger.
    PAGE_LIMIT = 50_000

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "MassiveAdapter requires the MASSIVE_API_KEY env var "
                "(loaded from secrets/secret.env) or an explicit "
                "api_key argument."
            )
        self._base_url = (
            base_url
            or os.environ.get("MASSIVE_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval not in _INTERVAL_MAP:
            raise ValueError(
                f"MassiveAdapter does not map interval={interval!r}. "
                f"Supported: {sorted(_INTERVAL_MAP)}"
            )
        multiplier, timespan = _INTERVAL_MAP[interval]
        # Polygon's ``to`` is inclusive, but only on whole-day buckets.
        # For sub-daily intervals it returns bars whose START is in
        # ``[from 00:00 ET, to 23:59 ET]``. To match the rest of the
        # codebase (which treats ``end`` as inclusive of the
        # caller-provided last day), we forward end + 1 day so the
        # final session's bars are guaranteed in the response, then
        # trim post-fetch.
        url = (
            f"{self._base_url}/v2/aggs/ticker/{symbol}/range/"
            f"{multiplier}/{timespan}/"
            f"{start.isoformat()}/{(end + timedelta(days=1)).isoformat()}"
        )
        rows = self._paginate(url)
        if not rows:
            raise ValueError(
                f"No data found for {symbol} between {start} and {end} "
                f"({interval}) [Massive]"
            )
        df = pd.DataFrame(rows)
        df.rename(
            columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
            },
            inplace=True,
        )
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(
            NY_TZ
        )
        out = df[["Open", "High", "Low", "Close", "Volume"]]
        if interval != "1d":
            # Trim the +1 day pad we added on the way in so callers
            # asking for "up to end" don't get bars from end+1.
            cutoff = pd.Timestamp(end + timedelta(days=1), tz=NY_TZ)
            out = out[out.index < cutoff]
        return out

    # ---- internals ----

    def _paginate(self, url: str) -> list[dict]:
        """Walk Polygon's ``next_url`` chain until exhausted.

        The first call carries the auth + sort + adjusted params; the
        ``next_url`` already encodes its own cursor so we just append
        the API key on each follow-up.
        """
        params: dict[str, object] = {
            "apiKey": self._api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": self.PAGE_LIMIT,
        }
        rows: list[dict] = []
        next_url: str | None = url
        first = True
        while next_url:
            resp = self._client.get(
                next_url, params=params if first else {"apiKey": self._api_key}
            )
            resp.raise_for_status()
            payload = resp.json()
            rows.extend(payload.get("results") or [])
            next_url = payload.get("next_url")
            first = False
        return rows
