"""AlphaVantageListingUniverseAdapter — CSV 파싱·필터·Mongo 캐시 (실 Mongo, 테스트 DB)."""

from __future__ import annotations

import os
import uuid
from datetime import date

import pytest
from pymongo import MongoClient

from data.adapters.alphavantage_listing_universe import (
    AlphaVantageListingUniverseAdapter,
    parse_universe_at,
)

_CSV = """symbol,name,exchange,assetType,ipoDate,delistingDate,status
AAPL,Apple Inc,NASDAQ,Stock,1980-12-12,null,Active
PCTY,Paylocity,NASDAQ,Stock,2014-03-19,null,Active
XLNX,Xilinx,NASDAQ,Stock,1990-06-12,null,Active
QQQ,Invesco QQQ,NASDAQ,ETF,1999-03-10,null,Active
BRK.B,Berkshire B,NYSE,Stock,1996-05-09,null,Active
XOM,Exxon,NYSE,Stock,1970-01-02,null,Active
ACY,AeroCentury,NYSE MKT,Stock,1998-01-02,null,Active
"""


class _Resp:
    def __init__(self, text, status_code=200):
        self.text, self.status_code = text, status_code


class _Http:
    def __init__(self, responses):
        self._r = list(responses); self.calls = 0

    def get(self, url, params=None):
        self.calls += 1
        return self._r.pop(0)


@pytest.fixture
def mongo():
    url = os.environ.get("MONGO_URL", "mongodb://mongo:27017")
    client = MongoClient(url, serverSelectionTimeoutMS=5000)
    db_name = f"test_listing_{uuid.uuid4().hex[:8]}"
    try:
        client.admin.command("ping")
    except Exception:
        pytest.skip("mongo unavailable")
    yield client, db_name
    client.drop_database(db_name)


def test_parse_universe_at():
    assert parse_universe_at("nasdaq_full@2020-01-02") == ("nasdaq_full", date(2020, 1, 2))
    assert parse_universe_at("NASDAQ_FULL@2020-01-02") == ("nasdaq_full", date(2020, 1, 2))
    assert parse_universe_at("nasdaq_full") is None
    assert parse_universe_at("nasdaq_full@garbage") is None


def test_filters_by_exchange_and_asset_type_and_caches_in_mongo(mongo):
    client, db = mongo
    http = _Http([_Resp(_CSV)])
    a = AlphaVantageListingUniverseAdapter(api_key="k", http_client=http, client=client, mongo_db=db)
    nas = a.get_tickers("nasdaq_full@2020-01-02")
    assert nas == ["AAPL", "PCTY", "XLNX"]                     # ETF 제외
    assert a.get_tickers("nyse_full@2020-01-02") == ["BRK-B", "XOM"]  # NYSE MKT 제외, '.'→'-'
    assert a.get_tickers("us_all@2020-01-02") == ["AAPL", "PCTY", "XLNX", "BRK-B", "XOM"]
    assert http.calls == 1                                      # 두 번째부터 Mongo 히트
    doc = client[db]["listing_status"].find_one({"source": "alphavantage", "date": "2020-01-02"})
    assert doc and len(doc["rows"]) == 7 and doc["rows"][1]["ipoDate"] == "2014-03-19"
    assert doc["rows"][0]["delistingDate"] is None


def test_non_dated_universe_is_not_ours(mongo):
    client, db = mongo
    a = AlphaVantageListingUniverseAdapter(api_key="k", http_client=_Http([]), client=client, mongo_db=db)
    with pytest.raises(ValueError):
        a.get_tickers("nasdaq_full")
    with pytest.raises(ValueError):
        a.get_tickers("kospi200@2020-01-02")
    with pytest.raises(ValueError):
        a.get_listing(date(2009, 12, 31))                       # 2010-01-01 이전 미지원


def test_rate_limit_json_retries_then_raises(mongo):
    client, db = mongo
    http = _Http([_Resp('{"Note": "rate limit"}'), _Resp(_CSV)])
    a = AlphaVantageListingUniverseAdapter(api_key="k", http_client=http, client=client, mongo_db=db,
                                           rate_limit_wait_s=0.0)
    assert a.get_tickers("nasdaq_full@2021-06-01") == ["AAPL", "PCTY", "XLNX"]
    assert http.calls == 2
