"""EODHD fundamentals adapter — float + splits.

REST endpoints used::

    GET /api/fundamentals/{symbol}.US?api_token=...
    GET /api/splits-dividends/{symbol}.US?api_token=...

The fundamentals payload is large and deeply nested; we only extract
two fields:

    SharesStats.SharesFloat       → public float (Ross's #1 criterion)
    SplitsDividends.NumberOfShares→ shares-outstanding history (unused)

Splits come from a separate endpoint (`/splits-dividends/`) which
returns ``[{"date":"YYYY-MM-DD","split":"X/Y"}, ...]``. We convert
the ``X/Y`` string to a ratio matching ``yfinance.Ticker.splits``
convention (>1 = forward, <1 = reverse).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd

from data.domain.ports import FundamentalsPort, TickerFundamentals


class EODHDFundamentalsAdapter(FundamentalsPort):
    """EODHD-backed fundamentals fetcher (float + splits)."""

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
                "EODHDFundamentalsAdapter requires the EODHD_API_KEY env "
                "var (loaded from secrets/secret.env) or an explicit "
                "api_key argument."
            )
        self._base_url = (
            base_url
            or os.environ.get("EODHD_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)

    def fetch(self, symbol: str) -> TickerFundamentals:
        eodhd_symbol = self._normalize_symbol(symbol)
        float_shares = self._fetch_float(eodhd_symbol)
        splits = self._fetch_splits(eodhd_symbol)
        return TickerFundamentals(
            symbol=symbol,
            float_shares=float_shares,
            splits=splits,
        )

    # ---- float ----

    def _fetch_float(self, eodhd_symbol: str) -> float | None:
        url = f"{self._base_url}/fundamentals/{eodhd_symbol}"
        params = {"api_token": self._api_key, "fmt": "json"}
        try:
            resp = self._client.get(url, params=params)
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        try:
            data: Any = resp.json()
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        # SharesStats.SharesFloat is the primary path. ETF / fund
        # responses lack the SharesStats block entirely → return None.
        stats = data.get("SharesStats") or {}
        f = stats.get("SharesFloat")
        if f is None:
            # Fallback to outstanding if float missing (some symbols
            # don't report float separately). Still better than nothing.
            f = stats.get("SharesOutstanding")
        if f is None:
            return None
        try:
            f = float(f)
        except (TypeError, ValueError):
            return None
        return f if f > 0 else None

    # ---- splits ----

    def _fetch_splits(self, eodhd_symbol: str) -> pd.Series | None:
        url = f"{self._base_url}/splits-dividends/{eodhd_symbol}"
        # Note: this is the same endpoint used for dividends; we only
        # consume the splits portion.
        params = {"api_token": self._api_key, "fmt": "json"}
        try:
            resp = self._client.get(url, params=params)
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except Exception:
            return None
        # The actual splits list lives under a different endpoint; the
        # ``/splits-dividends/`` root returns a summary blob. Use the
        # explicit splits-only path:
        return self._fetch_splits_explicit(eodhd_symbol)

    def _fetch_splits_explicit(self, eodhd_symbol: str) -> pd.Series | None:
        url = f"{self._base_url}/splits/{eodhd_symbol}"
        params = {"api_token": self._api_key, "fmt": "json"}
        try:
            resp = self._client.get(url, params=params)
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        try:
            rows = resp.json()
        except Exception:
            return None
        if not isinstance(rows, list) or not rows:
            return None
        # Each row: {"date": "YYYY-MM-DD", "split": "X/Y"} where the
        # ratio means "X new shares for Y old shares". yfinance
        # represents this as ``new/old`` — same convention.
        idx: list[pd.Timestamp] = []
        vals: list[float] = []
        for row in rows:
            try:
                d = pd.Timestamp(row["date"])
                ratio_str = str(row["split"])
                if "/" in ratio_str:
                    num, den = ratio_str.split("/", 1)
                    ratio = float(num) / float(den)
                else:
                    ratio = float(ratio_str)
                if ratio > 0:
                    idx.append(d)
                    vals.append(ratio)
            except (KeyError, ValueError, ZeroDivisionError):
                continue
        if not idx:
            return None
        s = pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()
        return s

    # ---- symbol mapping ----

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        # Mirror EODHDAdapter._normalize_symbol — append ``.US`` for
        # US tickers, leave KR/global suffixes intact.
        if "." in symbol:
            # Already exchange-tagged (e.g. ``005930.KS`` → keep KS path
            # for now; EODHD KR symbol mapping is .KO but fundamentals
            # for KR are out of scope for Bull Flag which is US-only).
            return symbol
        return f"{symbol}.US"
