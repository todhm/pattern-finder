"""Massive (Polygon) fundamentals adapter — float + splits.

Endpoints used::

    GET /v3/reference/tickers/{ticker}?apiKey=...
    GET /v3/reference/splits?ticker={ticker}&apiKey=...

Polygon doesn't expose "float" as a first-class field. The closest
proxy is ``weighted_shares_outstanding`` (= shares outstanding
weighted by class) which approximates float for single-class
issuers — accurate enough for the Bull Flag low-float gate (the
strategy compares against a 10M cap, not a precision-sensitive
threshold).

Splits live on a separate endpoint and return one row per event with
``split_from`` / ``split_to`` integers. Ratio convention matches
``yfinance.Ticker.splits`` (>1 = forward, <1 = reverse).
"""

from __future__ import annotations

import os

import httpx
import pandas as pd

from data.domain.ports import FundamentalsPort, TickerFundamentals


class MassiveFundamentalsAdapter(FundamentalsPort):
    """Polygon-backed fundamentals fetcher (float proxy + splits)."""

    DEFAULT_BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "MassiveFundamentalsAdapter requires MASSIVE_API_KEY env "
                "var (loaded from secrets/secret.env)."
            )
        self._base_url = (
            base_url
            or os.environ.get("MASSIVE_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)

    def fetch(self, symbol: str) -> TickerFundamentals:
        return TickerFundamentals(
            symbol=symbol,
            float_shares=self._fetch_float(symbol),
            splits=self._fetch_splits(symbol),
        )

    # ---- float ----

    def _fetch_float(self, ticker: str) -> float | None:
        url = f"{self._base_url}/v3/reference/tickers/{ticker}"
        params = {"apiKey": self._api_key}
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
        results = (data or {}).get("results") or {}
        # Try in order: weighted (float-like) → share-class outstanding
        # → total outstanding. Polygon's docs say weighted is closest
        # to public float for single-class issuers.
        for key in (
            "weighted_shares_outstanding",
            "share_class_shares_outstanding",
            "shares_outstanding",
        ):
            v = results.get(key)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
        return None

    # ---- splits ----

    def _fetch_splits(self, ticker: str) -> pd.Series | None:
        url = f"{self._base_url}/v3/reference/splits"
        params = {
            "ticker": ticker,
            "apiKey": self._api_key,
            "limit": 1000,  # plenty for any single ticker
        }
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
        rows = (data or {}).get("results") or []
        if not rows:
            return None
        idx: list[pd.Timestamp] = []
        vals: list[float] = []
        for row in rows:
            try:
                d = pd.Timestamp(row["execution_date"])
                # Polygon ratio: split_to / split_from new shares per
                # old share. Same convention as yfinance.
                num = float(row["split_to"])
                den = float(row["split_from"])
                if den == 0:
                    continue
                ratio = num / den
                if ratio > 0:
                    idx.append(d)
                    vals.append(ratio)
            except (KeyError, ValueError, ZeroDivisionError, TypeError):
                continue
        if not idx:
            return None
        s = pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()
        return s
