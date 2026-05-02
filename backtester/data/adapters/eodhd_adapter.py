"""EODHD (eodhistoricaldata.com) market-data adapter.

REST endpoint: ``GET https://eodhd.com/api/intraday/{symbol}`` for
sub-daily, ``/api/eod/{symbol}`` for daily and longer. Auth via the
``api_token`` query parameter; key loaded from the
``EODHD_API_KEY`` env var (set in ``secrets/secret.env``).

Capability matrix (verified against this account on 2026-05-01)::

    interval  history depth  KR KOSPI  KOSDAQ
    1m        2y+            ✓         likely ✓
    5m        2y+            ✓         likely ✓
    15m       10y+           ✓         likely ✓
    30m       2y+            ✓         likely ✓
    1h        ~2y            ✓         likely ✓
    1d        full           ✓         ✓

Symbol translation (yfinance convention → EODHD convention)::

    AAPL          → AAPL.US
    005930.KS     → 005930.KO    (yfinance KOSPI suffix is .KS,
                                  EODHD's is .KO with MIC XKRX)
    xxx.KQ        → xxx.KQ       (KOSDAQ matches in both)

Tz contract
    yfinance normalizes intraday frames to ``America/New_York``;
    we match here so the routing layer + cache + RTH filter all see
    a consistent index regardless of source. EODHD intraday returns
    a ``timestamp`` field (UTC unix epoch) and a ``datetime`` string
    (UTC); we use the unix value to avoid string-parsing ambiguity.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone

import httpx
import pandas as pd

from data.domain.ports import MarketDataPort

NY_TZ = "America/New_York"

# EODHD's intraday endpoint accepts only this exact set of strings.
# Anything outside (e.g. "10m", "2m") returns HTTP 422 "interval is
# invalid". We mirror yfinance's interval names here so the rest of
# the stack doesn't need to know about EODHD's vocabulary.
_INTRADAY_INTERVALS: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "1h",
    "1h": "1h",
}

# EODHD caps the per-request span on the intraday endpoint. 1-minute
# is the tightest at 120 days; coarser intervals get a wider window.
# Going past these limits returns ``HTTP 422 {"errors":{"to":["Max
# period length is 120 days"], ...}}``. We chunk requests that exceed
# the limit and stitch the dataframes together so callers can request
# arbitrary spans transparently. Values are conservative — a touch
# below EODHD's documented cap so off-by-one boundary timestamps
# don't trip the validator.
_INTRADAY_MAX_DAYS: dict[str, int] = {
    "1m": 120,
    "5m": 600,
    "15m": 600,
    "30m": 600,
    "60m": 600,
    "1h": 600,
}

# Daily / weekly / monthly through the /eod endpoint.
_EOD_PERIODS: dict[str, str] = {
    "1d": "d",
    "1wk": "w",
    "1mo": "m",
}


class EODHDAdapter(MarketDataPort):
    """Sub-daily + daily fetch implementation backed by EODHD."""

    DEFAULT_BASE_URL = "https://eodhd.com/api"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("EODHD_API_KEY")
        if not self._api_key:
            raise ValueError(
                "EODHDAdapter requires the EODHD_API_KEY env var "
                "(loaded from secrets/secret.env) or an explicit "
                "api_key argument."
            )
        self._base_url = (
            base_url
            or os.environ.get("EODHD_API_BASE")
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
        eodhd_symbol = self._normalize_symbol(symbol)
        if interval in _INTRADAY_INTERVALS:
            return self._fetch_intraday(
                eodhd_symbol, symbol, start, end, interval
            )
        if interval in _EOD_PERIODS:
            return self._fetch_eod(
                eodhd_symbol, symbol, start, end, interval
            )
        raise ValueError(
            f"EODHDAdapter does not map interval={interval!r}. "
            f"Supported intraday: {sorted(_INTRADAY_INTERVALS)}; "
            f"daily: {sorted(_EOD_PERIODS)}."
        )

    # ---- intraday ---------------------------------------------------

    def _fetch_intraday(
        self,
        eodhd_symbol: str,
        original_symbol: str,
        start: date,
        end: date,
        interval: str,
    ) -> pd.DataFrame:
        """Multi-chunk intraday fetch with EODHD's per-request window cap.

        EODHD's intraday endpoint rejects requests longer than its
        per-interval limit (1m → 120 days). We split the requested
        span into windows of ``_INTRADAY_MAX_DAYS[interval]`` and
        concatenate the per-window dataframes. Single-window requests
        cost the same as before; only oversized spans pay the
        extra round-trips.

        On the upper-side cache layer this is invisible — one cache
        miss → one parquet, regardless of how many sub-requests went
        out underneath.
        """
        max_days = _INTRADAY_MAX_DAYS.get(interval, 120)
        spans = list(_split_intraday_span(start, end, max_days))
        frames: list[pd.DataFrame] = []
        for span_start, span_end in spans:
            df_chunk = self._fetch_intraday_window(
                eodhd_symbol, original_symbol, span_start, span_end, interval
            )
            frames.append(df_chunk)
        # ``concat`` keeps the per-chunk DatetimeIndexes; the boundary
        # day is included in each adjacent chunk so a duplicate row
        # at exactly the cutover timestamp is possible. ``~duplicated``
        # drops the second copy. ``sort_index`` defends against any
        # out-of-order chunks (shouldn't happen, but cheap insurance).
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="first")].sort_index()
        return out

    def _fetch_intraday_window(
        self,
        eodhd_symbol: str,
        original_symbol: str,
        start: date,
        end: date,
        interval: str,
    ) -> pd.DataFrame:
        from_ts = int(
            datetime.combine(
                start, datetime.min.time(), tzinfo=timezone.utc
            ).timestamp()
        )
        # ``+1 day`` so the user's last calendar day is fully included
        # — EODHD's ``to`` is exclusive of the boundary timestamp.
        to_ts = int(
            datetime.combine(
                end + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ).timestamp()
        )
        url = f"{self._base_url}/intraday/{eodhd_symbol}"
        params = {
            "api_token": self._api_key,
            "interval": _INTRADAY_INTERVALS[interval],
            "from": from_ts,
            "to": to_ts,
            "fmt": "json",
        }
        resp = self._client.get(url, params=params)
        if resp.status_code != 200:
            raise ValueError(
                f"EODHD intraday {original_symbol} {interval}: "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            # Empty windows are fine when chunking — the symbol may
            # have had no trading on that span (holidays, weekend-
            # only chunks, etc.). Return an empty frame and let the
            # caller stitch.
            return pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"],
                index=pd.DatetimeIndex([], tz=NY_TZ),
            )
        df = pd.DataFrame(rows)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        df.index = pd.to_datetime(
            df["timestamp"], unit="s", utc=True
        ).dt.tz_convert(NY_TZ)
        out = df[["Open", "High", "Low", "Close", "Volume"]]
        # Drop placeholder bars EODHD inserts at the very end of the
        # KR session (15:00 KST single-tick close-auction reference
        # with ``V=None`` and OHLC all equal, plus a few similar
        # session-edge artifacts). Detector logic treats these as
        # legitimate flat candles and they break the "all-bullish"
        # FVG check + visualization. Two heuristics, both required:
        #   1. ``Volume`` is null  →  no real trading
        #   2. OHLC all identical  →  zero-range synthetic bar
        # Real low-volume bars on liquid names still carry an OHLC
        # range so this filter doesn't accidentally drop them.
        volume_missing = out["Volume"].isna()
        zero_range = (
            (out["Open"] == out["High"])
            & (out["High"] == out["Low"])
            & (out["Low"] == out["Close"])
        )
        out = out[~(volume_missing & zero_range)]
        return out

    # ---- daily / weekly / monthly ----------------------------------

    def _fetch_eod(
        self,
        eodhd_symbol: str,
        original_symbol: str,
        start: date,
        end: date,
        interval: str,
    ) -> pd.DataFrame:
        url = f"{self._base_url}/eod/{eodhd_symbol}"
        params = {
            "api_token": self._api_key,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "period": _EOD_PERIODS[interval],
            "fmt": "json",
        }
        resp = self._client.get(url, params=params)
        if resp.status_code != 200:
            raise ValueError(
                f"EODHD eod {original_symbol} {interval}: "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"No data found for {original_symbol} between "
                f"{start} and {end} ({interval}) [EODHD]"
            )
        df = pd.DataFrame(rows)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        # Daily index is tz-naive midnight to match yfinance daily
        # convention (callers like the FVG detector check tz-aware
        # vs naive to decide whether the frame is intraday).
        df.index = pd.to_datetime(df["date"])
        return df[["Open", "High", "Low", "Close", "Volume"]]

    # ---- symbol translation ----------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Translate a yfinance-shaped ticker into EODHD's format.

        - Bare ``AAPL`` (US) gets ``.US`` appended.
        - ``005930.KS`` (yfinance KOSPI suffix) → ``005930.KO`` (EODHD
          uses ``.KO`` with MIC ``XKRX`` for KOSPI).
        - ``xxx.KQ`` (KOSDAQ) is unchanged — EODHD also uses ``.KQ``.
        - Anything else with a dot is passed through verbatim so users
          can hand-write EODHD-native symbols (e.g., ``BMW.XETRA``).
        """
        upper = symbol.upper()
        if "." not in upper:
            return f"{upper}.US"
        if upper.endswith(".KS"):
            return upper[:-3] + ".KO"
        return upper


def _split_intraday_span(
    start: date, end: date, max_days: int
) -> "list[tuple[date, date]]":
    """Split ``[start, end]`` (inclusive) into windows of ≤ ``max_days``.

    The span boundary is intentionally exclusive on the upstream side
    — EODHD's ``to`` parameter already treats the next-day boundary
    as exclusive, so chunking by adding ``timedelta(days=max_days)``
    produces non-overlapping native windows. We add 1 day overlap on
    the rejoin path (``concat`` + ``drop_duplicates``) to defend
    against off-by-one boundary drift.
    """
    spans: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=max_days - 1))
        spans.append((cursor, chunk_end))
        if chunk_end == end:
            break
        cursor = chunk_end + timedelta(days=1)
    return spans
