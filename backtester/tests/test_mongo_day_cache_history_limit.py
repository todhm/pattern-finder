"""MongoDayCacheAdapter — 서브데일리 히스토리 한계 보호 회귀 테스트.

yfinance 15m처럼 최근 N일만 주는 소스가 오래된 날짜를 '휴장 빈
문서'로 영구 캐시해 다른 소스의 백필을 막던 버그(2026-08)의 재발
방지. 실제 mongo 컨테이너를 쓰고 테스트 전용 DB를 만들었다 지운다.
"""

from __future__ import annotations

import os
import uuid
from datetime import date

import pandas as pd
import pytest
from pymongo import MongoClient

from data.adapters.mongo_day_cache import MongoDayCacheAdapter
from data.domain.ports import MarketDataPort

NY = "America/New_York"


class _LimitedHistoryStub(MarketDataPort):
    """요청 구간과 무관하게 2024-06-24 이후 봉만 반환하는 업스트림."""

    def __init__(self):
        self.calls = 0

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        self.calls += 1
        idx = pd.DatetimeIndex(
            ["2024-06-24 09:30", "2024-06-25 09:30"], tz=NY
        )
        return pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.0, 2.0],
                "Low": [1.0, 2.0],
                "Close": [1.0, 2.0],
                "Volume": [10.0, 10.0],
            },
            index=idx,
        )


class _EmptyStub(MarketDataPort):
    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"]
        )


@pytest.fixture
def mongo_db():
    url = os.environ.get("MONGO_URL", "mongodb://mongo:27017")
    db_name = f"test_day_cache_{uuid.uuid4().hex[:8]}"
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("mongo unavailable")
    yield client, db_name
    client.drop_database(db_name)


def _adapter(upstream, mongo_db):
    client, db_name = mongo_db
    return MongoDayCacheAdapter(
        upstream, source_name="stub", mongo_db=db_name, client=client
    )


class TestHistoryLimitGuard:
    def test_days_before_first_bar_are_not_cached_empty(self, mongo_db):
        client, db_name = mongo_db
        cache = _adapter(_LimitedHistoryStub(), mongo_db)
        df = cache.fetch_ohlcv(
            "TQQQ", date(2024, 6, 17), date(2024, 6, 25), interval="15m"
        )
        assert len(df) == 2
        coll = client[db_name]["bars_stub"]
        # 첫 봉(6/24) 이전 평일(6/17~21)은 문서가 없어야 한다.
        assert coll.count_documents({"date": "2024-06-18"}) == 0
        # 반환된 날은 정상 캐시.
        assert coll.count_documents({"date": "2024-06-24"}) == 1

    def test_uncovered_days_stay_missing_for_other_sources(self, mongo_db):
        cache = _adapter(_LimitedHistoryStub(), mongo_db)
        cache.fetch_ohlcv(
            "TQQQ", date(2024, 6, 17), date(2024, 6, 25), interval="15m"
        )
        # peek은 미커버 날짜 때문에 None → 폴백 체인이 다음 소스 시도.
        assert (
            cache.peek_cache(
                "TQQQ", date(2024, 6, 17), date(2024, 6, 25), interval="15m"
            )
            is None
        )

    def test_all_empty_sub_daily_writes_nothing(self, mongo_db):
        client, db_name = mongo_db
        cache = _adapter(_EmptyStub(), mongo_db)
        df = cache.fetch_ohlcv(
            "TQQQ", date(2022, 6, 1), date(2022, 6, 10), interval="15m"
        )
        assert df.empty
        assert client[db_name]["bars_stub"].count_documents({}) == 0

    def test_daily_interval_keeps_holiday_caching(self, mongo_db):
        client, db_name = mongo_db
        cache = _adapter(_LimitedHistoryStub(), mongo_db)
        cache.fetch_ohlcv(
            "TQQQ", date(2024, 6, 17), date(2024, 6, 25), interval="1d"
        )
        # 일봉은 기존 동작: 미반환 날도 빈 성공 문서 (휴장 처리).
        coll = client[db_name]["bars_stub"]
        assert coll.count_documents({"date": "2024-06-18"}) == 1


class TestNoDataTTL:
    """'No data found' 실패는 90일 TTL — 24h 마다 재조회하지 않는다."""

    def test_nodata_fail_marker_stays_fresh_after_one_day(self, mongo_db):
        from datetime import datetime, timedelta, timezone

        client, db_name = mongo_db

        class _NoData(MarketDataPort):
            def __init__(self): self.calls = 0
            def fetch_ohlcv(self, symbol, start, end, interval="1d"):
                self.calls += 1
                raise ValueError(f"No data found for {symbol} between {start} and {end} ({interval})")

        up = _NoData()
        cache = MongoDayCacheAdapter(up, source_name=f"t_{uuid.uuid4().hex[:6]}",
                                     client=client, mongo_db=db_name)
        with pytest.raises(Exception):
            cache.fetch_ohlcv("ARM", date(2019, 1, 2), date(2019, 1, 10))
        assert up.calls == 1
        # 마커를 2일 전으로 되돌려도 (24h TTL 이면 만료) nodata 는 아직 fresh
        cache._bars_coll.update_many({"symbol": "ARM"}, {"$set": {
            "fetched_at": datetime.now(timezone.utc) - timedelta(days=2)}})
        with pytest.raises(Exception):
            cache.fetch_ohlcv("ARM", date(2019, 1, 2), date(2019, 1, 10))
        assert up.calls == 1                                    # 네트워크 재호출 없음
        # 91일 전이면 만료 → 재조회
        cache._bars_coll.update_many({"symbol": "ARM"}, {"$set": {
            "fetched_at": datetime.now(timezone.utc) - timedelta(days=91)}})
        with pytest.raises(Exception):
            cache.fetch_ohlcv("ARM", date(2019, 1, 2), date(2019, 1, 10))
        assert up.calls == 2
