"""IPO 캘린더 composition — Finnhub(공모가 포함) → Alpha Vantage LISTING_STATUS 폴백.

페이지·CLI 는 :func:`build_default_ipo_calendar` 만 호출한다. 키가 없어 어댑터
생성이 실패하면 그 소스는 조용히 빠지고, 하나도 없으면 ``None`` — 호출자가
수동 티커 입력으로 안내한다.
"""

from __future__ import annotations

from datetime import date

from data.domain.models import IpoEvent
from data.domain.ports import IpoCalendarPort


class FallbackIpoCalendarAdapter(IpoCalendarPort):
    """앞 소스가 빈 결과(또는 예외)면 다음 소스. 첫 비어 있지 않은 결과를 돌려준다."""

    def __init__(self, sources: list[tuple[str, IpoCalendarPort]]) -> None:
        if not sources:
            raise ValueError("at least one IPO calendar source is required")
        self._sources = sources
        self.last_source: str | None = None

    def fetch_ipos(self, start: date, end: date) -> list[IpoEvent]:
        for label, src in self._sources:
            try:
                out = src.fetch_ipos(start, end)
            except Exception:  # noqa: BLE001
                out = []
            if out:
                self.last_source = label
                return out
        self.last_source = None
        return []


def build_default_ipo_calendar() -> FallbackIpoCalendarAdapter | None:
    sources: list[tuple[str, IpoCalendarPort]] = []
    try:
        from data.adapters.finnhub_ipo_calendar import FinnhubIpoCalendarAdapter
        sources.append(("Finnhub", FinnhubIpoCalendarAdapter()))
    except Exception:  # noqa: BLE001 — 키/Mongo 없음
        pass
    try:
        from data.adapters.alphavantage_listing_ipos import AlphaVantageListingIposAdapter
        sources.append(("AlphaVantage LISTING_STATUS", AlphaVantageListingIposAdapter()))
    except Exception:  # noqa: BLE001
        pass
    return FallbackIpoCalendarAdapter(sources) if sources else None
