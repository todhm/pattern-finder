"""cache_frontier_fill — 날짜 단위 캐시 앞단을 전 종목 1콜/일로 채우는 로직.

실제 mongo 컨테이너에 테스트 전용 DB 를 만들었다 지운다 (history_limit 테스트와 동일).
"""

from __future__ import annotations

import os
import uuid
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from pymongo import MongoClient

from data.adapters.cache_frontier_fill import fill_recent_days
from data.adapters.mongo_day_cache import MongoDayCacheAdapter
from data.domain.ports import BulkDailyBarsPort, MarketDataPort


class _Upstream(MarketDataPort):
    def __init__(self):
        self.calls: list[str] = []

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        self.calls.append(symbol)
        idx = pd.bdate_range(start, end)
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)


class _Bulk(BulkDailyBarsPort):
    def __init__(self, frames: dict[date, pd.DataFrame], splits: list[str] = ()):
        self.frames = frames
        self.splits = list(splits)
        self.calls: list[date] = []

    def fetch_day(self, day):
        self.calls.append(day)
        if day not in self.frames:
            raise ValueError("boom")
        return self.frames[day]

    def split_symbols(self, start, end):
        return self.splits


def _frame(symbols, px=10.0):
    return pd.DataFrame({"Open": px, "High": px, "Low": px, "Close": px, "Volume": 100.0},
                        index=pd.Index(symbols, name="symbol"))


@pytest.fixture
def mongo_db():
    url = os.environ.get("MONGO_URL", "mongodb://mongo:27017")
    db_name = f"test_fill_{uuid.uuid4().hex[:8]}"
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("mongo unavailable")
    yield client, url, db_name
    client.drop_database(db_name)


def test_fill_makes_per_ticker_fetch_a_cache_hit(mongo_db):
    client, url, db_name = mongo_db
    up = _Upstream()
    cache = MongoDayCacheAdapter(up, "t", mongo_url=url, mongo_db=db_name, client=client)
    # 2026-10-13(화)~16(금) 거래일, 오늘은 10-18(일). 10-15 는 휴장으로 가정.
    today = date(2026, 10, 18)
    syms = ["AAA", "BBB"]
    bulk = _Bulk({
        date(2026, 10, 13): _frame(syms), date(2026, 10, 14): _frame(syms),
        date(2026, 10, 15): pd.DataFrame(), date(2026, 10, 16): _frame(syms),
        **{d.date(): _frame(syms) for d in pd.bdate_range("2026-09-22", "2026-10-12")},
    })
    rep = fill_recent_days(cache, bulk, syms, date(2026, 10, 18), lookback_bdays=4, today=today)
    assert rep.days_filled == [date(2026, 10, 13), date(2026, 10, 14), date(2026, 10, 16)]
    assert rep.days_holiday == [date(2026, 10, 15)]
    assert rep.docs_written == 8 and not rep.errors
    df = cache.fetch_ohlcv("AAA", date(2026, 10, 13), date(2026, 10, 16))
    assert up.calls == []  # 전부 캐시 적중
    assert len(df) == 3 and float(df["Close"].iloc[-1]) == 10.0
    # 두 번째 실행: 커버리지 충족 → bulk 호출 없음
    bulk.calls.clear()
    rep2 = fill_recent_days(cache, bulk, syms, date(2026, 10, 18), lookback_bdays=4, today=today)
    assert bulk.calls == [date(2026, 10, 16)]  # 최신일 1콜(분모)만, 나머지는 커버리지로 생략
    assert rep2.days_filled == [] and len(rep2.days_skipped_covered) == 4


def test_dead_symbols_do_not_block_coverage(mongo_db):
    client, url, db_name = mongo_db
    cache = MongoDayCacheAdapter(_Upstream(), "t", mongo_url=url, mongo_db=db_name, client=client)
    today = date(2026, 10, 18)
    alive = ["AAA", "BBB", "CCC"]
    syms = alive + ["DEAD1", "DEAD2"]  # 유니버스의 40% 가 두 소스 모두 모르는 심볼
    bulk = _Bulk({d.date(): _frame(alive) for d in pd.bdate_range("2026-10-13", "2026-10-16")})
    rep = fill_recent_days(cache, bulk, syms, date(2026, 10, 18), lookback_bdays=4, today=today)
    assert len(rep.days_filled) == 4 and rep.docs_written == 12
    bulk.calls.clear()
    rep2 = fill_recent_days(cache, bulk, syms, date(2026, 10, 18), lookback_bdays=4, today=today)
    assert bulk.calls == [date(2026, 10, 16)] and rep2.days_filled == []


def test_today_and_leading_empty_days_are_not_written(mongo_db):
    client, url, db_name = mongo_db
    cache = MongoDayCacheAdapter(_Upstream(), "t", mongo_url=url, mongo_db=db_name, client=client)
    today = date(2026, 10, 16)
    bulk = _Bulk({date(2026, 10, 14): pd.DataFrame(), date(2026, 10, 15): pd.DataFrame()})
    rep = fill_recent_days(cache, bulk, ["AAA"], today, lookback_bdays=2, today=today)
    # 오늘(10-16)은 제외, 빈 응답만 있으면 휴장 확정 불가 → 보류, 아무것도 안 씀
    assert rep.days_checked == [date(2026, 10, 14), date(2026, 10, 15)]
    assert rep.days_deferred == [date(2026, 10, 14), date(2026, 10, 15)]
    assert rep.docs_written == 0
    assert cache.coverage(date(2026, 10, 15), ["AAA"]) == 0


def test_split_symbol_cache_is_dropped_and_refetched(mongo_db):
    client, url, db_name = mongo_db
    up = _Upstream()
    cache = MongoDayCacheAdapter(up, "t", mongo_url=url, mongo_db=db_name, client=client)
    # 분할 전(10-01)에 받은 문서를 심어 둔다
    cache.fetch_ohlcv("SPL", date(2026, 10, 1), date(2026, 10, 9))
    client[db_name]["bars_t"].update_many({"symbol": "SPL"}, {"$set": {
        "fetched_at": datetime(2026, 10, 1, tzinfo=timezone.utc)}})
    up.calls.clear()
    today = date(2026, 10, 18)
    syms = ["SPL", "AAA"]
    bulk = _Bulk({d.date(): _frame(syms) for d in pd.bdate_range("2026-10-12", "2026-10-16")},
                 splits=["SPL"])
    rep = fill_recent_days(cache, bulk, syms, date(2026, 10, 16), lookback_bdays=5, today=today)
    assert rep.splits_dropped == {"SPL": 7}
    cache.fetch_ohlcv("SPL", date(2026, 10, 1), date(2026, 10, 16))
    assert up.calls == ["SPL"]  # 종목별 소스로 전체 재조회
    cache.fetch_ohlcv("AAA", date(2026, 10, 12), date(2026, 10, 16))
    assert up.calls == ["SPL"]  # AAA 는 채운 문서로 적중


def test_fill_day_skips_symbols_missing_from_frame(mongo_db):
    client, url, db_name = mongo_db
    cache = MongoDayCacheAdapter(_Upstream(), "t", mongo_url=url, mongo_db=db_name, client=client)
    n = cache.fill_day(date(2026, 10, 16), _frame(["AAA"]), ["AAA", "DEAD"])
    assert n == 1
    assert cache.coverage(date(2026, 10, 16), ["AAA", "DEAD"]) == 1


def test_dead_symbol_gets_nodata_markers_in_every_chain_cache(mongo_db):
    client, url, db_name = mongo_db
    up_a, up_b = _Upstream(), _Upstream()
    cache_a = MongoDayCacheAdapter(up_a, "a", mongo_url=url, mongo_db=db_name, client=client)
    cache_b = MongoDayCacheAdapter(up_b, "b", mongo_url=url, mongo_db=db_name, client=client)
    today = date(2026, 10, 18)
    # DEAD 는 10-13 에 소스 a 가 이미 No data found 를 기록한 상태. NEW 는 처음 보는 심볼.
    client[db_name]["bars_a"].insert_one({
        "symbol": "DEAD", "interval": "1d", "date": "2026-10-13", "status": "fail",
        "error_type": "ValueError", "error_msg": "No data found for DEAD between ...",
        "fetched_at": datetime(2026, 10, 14, tzinfo=timezone.utc)})
    syms = ["AAA", "DEAD", "NEW"]
    bulk = _Bulk({d.date(): _frame(["AAA"]) for d in pd.bdate_range("2026-10-13", "2026-10-16")})
    rep = fill_recent_days(cache_a, bulk, syms, date(2026, 10, 16), lookback_bdays=4, today=today,
                           marker_caches=[cache_a, cache_b])
    # DEAD: 4일 × 2캐시 = 8 마커 (10-13 은 a 에 이미 fail 문서 → 덮어씀, 카운트 포함)
    assert rep.dead_marked == 8
    with pytest.raises(Exception):
        cache_a.fetch_ohlcv("DEAD", date(2026, 10, 13), date(2026, 10, 16))
    with pytest.raises(Exception):
        cache_b.fetch_ohlcv("DEAD", date(2026, 10, 13), date(2026, 10, 16))
    assert up_a.calls == [] and up_b.calls == []  # 업스트림 호출 없이 즉시 실패
    # NEW 는 증거가 하나뿐 → 마커 없음, 종목별 경로가 직접 확인
    cache_a.fetch_ohlcv("NEW", date(2026, 10, 13), date(2026, 10, 16))
    assert up_a.calls == ["NEW"]
