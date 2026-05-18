"""EODHD news catalyst adapter.

REST endpoint: ``GET https://eodhd.com/api/news``

Capability (verified 2026-05):

    Per-symbol filter:  ``s=TSLA.US``  (single ticker)
    Date range:         ``from=YYYY-MM-DD&to=YYYY-MM-DD``
    Pagination:         ``offset / limit``  (1000 max per request)
    Sentiment:          included as ``sentiment.polarity`` ∈ [-1, +1]
                        when available.

Strategies use the date-of-publication only (Matt Diamond gate: "trade
when there's a news catalyst in the last N days"). Sentiment is
exposed via :class:`NewsEvent.sentiment` so callers can layer a "only
positive sentiment" filter on top.

Failure contract: returns ``[]`` on HTTP error / empty response. Same
graceful-degradation pattern as :class:`EODHDEarningsAdapter`.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import date, datetime, timezone

import httpx

from data.domain.models import NewsEvent
from data.domain.ports import NewsCatalystPort

log = logging.getLogger(__name__)

# EODHD's news endpoint accepts up to 1000 items per request. For a
# busy ticker over a year, the total may exceed that — we paginate by
# offset until the response is shorter than the page size or we hit
# the requested end-date.
_PAGE_SIZE = 1000
_MAX_PAGES = 10  # hard ceiling so a misconfigured query can't infinite-loop


class EODHDNewsAdapter(NewsCatalystPort):
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
                "EODHDNewsAdapter requires the EODHD_API_KEY env var "
                "or an explicit api_key argument."
            )
        self._base_url = (
            base_url
            or os.environ.get("EODHD_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)

    # ---- public API ---------------------------------------------------

    def fetch_news(
        self,
        symbol: str,
        start: date,
        end: date,
        limit_per_day: int | None = None,
    ) -> list[NewsEvent]:
        eodhd_symbol = self._normalize_symbol(symbol)
        url = f"{self._base_url}/news"
        all_rows: list[dict] = []
        offset = 0
        for _page in range(_MAX_PAGES):
            params = {
                "api_token": self._api_key,
                "s": eodhd_symbol,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "limit": _PAGE_SIZE,
                "offset": offset,
                "fmt": "json",
            }
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code != 200:
                    log.warning(
                        "EODHD news %s: HTTP %s %s",
                        symbol, resp.status_code, resp.text[:200],
                    )
                    break
                payload = resp.json()
            except Exception as exc:
                log.warning("EODHD news %s: %s", symbol, exc)
                break
            if not isinstance(payload, list) or not payload:
                break
            all_rows.extend(payload)
            if len(payload) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

        # Parse rows -> NewsEvent. EODHD shape per row:
        #   {"date": "2026-04-23T22:15:00+00:00",
        #    "title": "...", "content": "...",
        #    "symbols": ["TSLA.US", ...],
        #    "sentiment": {"polarity": 0.12, "neg":..., "neu":..., "pos":...}}
        out: list[NewsEvent] = []
        for row in all_rows:
            try:
                raw_dt = row.get("date")
                if not raw_dt:
                    continue
                published = _parse_iso(raw_dt)
                if published is None:
                    continue
                if published.date() < start or published.date() > end:
                    continue
                title = row.get("title") or ""
                pol = None
                sent = row.get("sentiment")
                if isinstance(sent, dict):
                    pol = _safe_float(sent.get("polarity"))
                out.append(
                    NewsEvent(
                        symbol=symbol,
                        published_at=published,
                        title=str(title),
                        sentiment=pol,
                    )
                )
            except Exception:
                continue
        out.sort(key=lambda n: n.published_at)

        # Per-day cap (preserve oldest items in the day so deterministic).
        if limit_per_day is not None and limit_per_day > 0:
            counts: defaultdict[date, int] = defaultdict(int)
            capped: list[NewsEvent] = []
            for ev in out:
                d = ev.published_at.date()
                if counts[d] >= limit_per_day:
                    continue
                counts[d] += 1
                capped.append(ev)
            return capped
        return out

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if "." in symbol:
            if symbol.endswith(".KS"):
                return symbol[:-3] + ".KO"
            return symbol
        return f"{symbol}.US"


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_iso(s: str) -> datetime | None:
    """Parse ISO-8601 strings EODHD returns ("...+00:00" or "...Z").
    Returns a tz-aware UTC datetime.
    """
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
