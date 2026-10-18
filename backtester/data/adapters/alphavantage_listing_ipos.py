"""Alpha Vantage LISTING_STATUS 의 ``ipoDate`` 로 만든 IPO 캘린더 — Finnhub 폴백.

:class:`AlphaVantageListingUniverseAdapter.get_listing` (기준일 활성 종목 전체,
Mongo 영구 캐시) 를 여러 기준일에서 합집합으로 모아 ``ipoDate`` 가 구간 안인
종목을 돌려준다. 스냅샷은 ``end`` 부터 ``step_days`` 씩 거슬러 올라가며 찍는다 —
``end`` 시점엔 이미 폐지된 IPO 도 그 전 스냅샷에는 살아 있으므로 생존 편향이
줄어든다 (완전 제거는 아님: 두 스냅샷 사이에 상장·폐지된 종목은 못 본다).

공모가·주식수는 없다 (``offer_price=None``). ``ipoDate`` 는 거래소 상장일이라
스핀오프·업리스팅도 섞인다 — 소비자가 첫 거래일 OHLCV 로 교차 확인한다.
"""

from __future__ import annotations

from datetime import date, timedelta

from data.adapters.alphavantage_listing_universe import AlphaVantageListingUniverseAdapter
from data.domain.models import IpoEvent
from data.domain.ports import IpoCalendarPort


class AlphaVantageListingIposAdapter(IpoCalendarPort):
    def __init__(self, listing: AlphaVantageListingUniverseAdapter | None = None, step_days: int = 182) -> None:
        self._listing = listing or AlphaVantageListingUniverseAdapter()
        self._step = step_days

    def fetch_ipos(self, start: date, end: date) -> list[IpoEvent]:
        start = max(start, self._listing.MIN_DATE)
        snap_end = min(end + timedelta(days=self._step), date.today())
        snaps: list[date] = []
        d = snap_end
        while d >= start:
            snaps.append(d)
            d -= timedelta(days=self._step)
        events: dict[str, IpoEvent] = {}
        for s in snaps:
            try:
                rows = self._listing.get_listing(s)
            except Exception:  # noqa: BLE001 — 한 스냅샷 실패는 건너뛴다
                continue
            for r in rows:
                if r.get("assetType") != "Stock" or not r.get("ipoDate"):
                    continue
                try:
                    ipo = date.fromisoformat(str(r["ipoDate"])[:10])
                except ValueError:
                    continue
                if not (start <= ipo <= end):
                    continue
                sym = (r.get("symbol") or "").strip().replace(".", "-")
                if not sym or sym in events:
                    continue
                events[sym] = IpoEvent(symbol=sym, name=r.get("name"), ipo_date=ipo, exchange=r.get("exchange"),
                                       status="priced", source="alphavantage_listing")
        return sorted(events.values(), key=lambda e: (e.ipo_date, e.symbol))
