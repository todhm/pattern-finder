"""EODHD earnings-calendar adapter.

REST endpoint: ``GET https://eodhd.com/api/calendar/earnings``

Capability (verified 2026-05 against this account):

    Per-symbol filter:  ``symbols=TSLA.US``
    Date range:         ``from=YYYY-MM-DD&to=YYYY-MM-DD``
    History:            ~5 years back, full forward window.
    Fields used:        ``report_date``, ``before_after_market``,
                        ``actual``, ``estimate``.

Symbol translation mirrors :class:`EODHDAdapter` (``AAPL`` → ``AAPL.US``,
``005930.KS`` → ``005930.KO``).

Failure contract: returns ``[]`` on HTTP error / empty response /
parse failure. Strategies that depend on this data should gate on
``len(events) > 0`` before activating the earnings-window check —
gracefully degrade instead of crashing the page.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime

import httpx

from data.domain.models import EarningsEvent
from data.domain.ports import EarningsCalendarPort

log = logging.getLogger(__name__)


class EODHDEarningsAdapter(EarningsCalendarPort):
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
                "EODHDEarningsAdapter requires the EODHD_API_KEY env "
                "var or an explicit api_key argument."
            )
        self._base_url = (
            base_url
            or os.environ.get("EODHD_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)

    # ---- public API ---------------------------------------------------

    def fetch_earnings(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> list[EarningsEvent]:
        eodhd_symbol = self._normalize_symbol(symbol)
        url = f"{self._base_url}/calendar/earnings"
        params = {
            "api_token": self._api_key,
            "symbols": eodhd_symbol,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "fmt": "json",
        }
        try:
            resp = self._client.get(url, params=params)
            if resp.status_code != 200:
                log.warning(
                    "EODHD earnings %s: HTTP %s %s",
                    symbol, resp.status_code, resp.text[:200],
                )
                return []
            payload = resp.json()
        except Exception as exc:
            log.warning("EODHD earnings %s: %s", symbol, exc)
            return []

        # Response shape: {"earnings": [{"code": "TSLA.US", "report_date":
        #   "2026-04-23", "date": "2026-Q1", "before_after_market": "AMC",
        #   "currency": "USD", "actual": 0.73, "estimate": 0.60, ...},
        #   ...]}
        rows = payload.get("earnings") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        out: list[EarningsEvent] = []
        for row in rows:
            try:
                rep = row.get("report_date") or row.get("date")
                if not rep:
                    continue
                rep_dt = datetime.strptime(rep[:10], "%Y-%m-%d").date()
                if rep_dt < start or rep_dt > end:
                    continue
                out.append(
                    EarningsEvent(
                        symbol=symbol,
                        report_date=rep_dt,
                        before_after_market=row.get("before_after_market"),
                        eps_actual=_safe_float(row.get("actual")),
                        eps_estimate=_safe_float(row.get("estimate")),
                    )
                )
            except Exception:
                continue
        out.sort(key=lambda e: e.report_date)
        return out

    # ---- helpers ------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """yfinance → EODHD convention (mirrors EODHDAdapter)."""
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
