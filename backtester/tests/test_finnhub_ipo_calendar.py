"""FinnhubIpoCalendarAdapter — 월 페이징·파싱·Mongo 월 캐시 (실 Mongo, 테스트 DB) + composite 폴백."""

from __future__ import annotations

import os
import uuid
from datetime import date

import pytest
from pymongo import MongoClient

from data.adapters.composed_ipo_calendar import FallbackIpoCalendarAdapter
from data.adapters.finnhub_ipo_calendar import FinnhubIpoCalendarAdapter, _month_starts
from data.domain.models import IpoEvent


class _Resp:
    def __init__(self, payload, status_code=200):
        self._p, self.status_code = payload, status_code

    def json(self):
        return self._p


class _Http:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def get(self, url, params=None):
        self.calls.append((params["from"], params["to"]))
        frm = params["from"]
        rows = [
            {"date": f"{frm[:7]}-15", "exchange": "NASDAQ Global", "name": f"Co {frm[:7]}", "numberOfShares": 1000000,
             "price": "12.50", "status": "priced", "symbol": f"S{frm[5:7]}", "totalSharesValue": 12500000},
            {"date": f"{frm[:7]}-20", "exchange": None, "name": "Withdrawn Inc", "numberOfShares": None,
             "price": None, "status": "withdrawn", "symbol": None, "totalSharesValue": None},
            {"date": f"{frm[:7]}-21", "exchange": "NYSE", "name": "Dot Class", "numberOfShares": None,
             "price": "", "status": "priced", "symbol": "ABC.A", "totalSharesValue": None},
        ]
        return _Resp({"ipoCalendar": rows})


@pytest.fixture
def mongo():
    url = os.environ.get("MONGO_URL", "mongodb://mongo:27017")
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    db_name = f"test_ipo_{uuid.uuid4().hex[:8]}"
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("mongo unavailable")
    yield client, db_name
    client.drop_database(db_name)


def test_month_starts():
    assert _month_starts(date(2021, 11, 10), date(2022, 2, 3)) == [date(2021, 11, 1), date(2021, 12, 1), date(2022, 1, 1), date(2022, 2, 1)]


def test_fetch_pages_by_month_parses_and_caches(mongo):
    client, db = mongo
    http = _Http()
    a = FinnhubIpoCalendarAdapter(api_key="k", http_client=http, client=client, mongo_db=db)
    out = a.fetch_ipos(date(2021, 1, 10), date(2021, 3, 25))
    assert len(http.calls) == 3
    syms = [e.symbol for e in out]
    assert syms == ["S01", "ABC-A", "S02", "S03"]     # 날짜순, withdrawn 제외, 1월 ABC.A → ABC-A
    e = out[0]
    assert e.ipo_date == date(2021, 1, 15) and e.offer_price == 12.5 and e.shares == 1e6 and e.source == "finnhub"
    assert e.exchange_group == "NASDAQ" and out[1].offer_price is None
    # 같은 구간 재조회 → 과거 달은 Mongo 히트, 네트워크 없음
    a.fetch_ipos(date(2021, 1, 1), date(2021, 3, 31))
    assert len(http.calls) == 3


def test_fallback_uses_next_source_when_first_empty():
    class _Empty:
        def fetch_ipos(self, s, e):
            return []

    class _One:
        def fetch_ipos(self, s, e):
            return [IpoEvent(symbol="Z", ipo_date=date(2020, 5, 5), source="x")]

    f = FallbackIpoCalendarAdapter([("empty", _Empty()), ("one", _One())])
    assert [e.symbol for e in f.fetch_ipos(date(2020, 1, 1), date(2020, 12, 31))] == ["Z"]
    assert f.last_source == "one"
