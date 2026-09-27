"""Point-in-time 유니버스 — Alpha Vantage ``LISTING_STATUS`` + MongoDB 캐시.

``nasdaq_full@2020-01-02`` 처럼 **유니버스 이름에 기준일을 붙이면** 그날
상장돼 있던 종목 목록을 준다 (생존 편향 제거). 현재 리스트(nasdaqtrader)
는 그 사이 상장된 종목이 섞이고 폐지·인수된 종목은 빠져 있어, 과거 기준일
스크린에서는 결과가 실제보다 좋게 나온다.

소스: ``GET https://www.alphavantage.co/query?function=LISTING_STATUS
&date=YYYY-MM-DD&state=active`` → CSV
``symbol,name,exchange,assetType,ipoDate,delistingDate,status``.
2010-01-01 이후 날짜만 지원. 하루치 전체(≈8,700행)가 한 번에 온다.

저장: Mongo ``listing_status`` 컬렉션, **날짜당 문서 1개**::

    { source: "alphavantage", date: "2020-01-02", fetched_at,
      rows: [{symbol, name, exchange, assetType, ipoDate, delistingDate}, ...] }

과거 사실이라 TTL 없이 영구 보관. (요청: 디스크 파일 대신 Mongo — 컨테이너
재빌드에도 남고 다른 머신과 공유.)

지원 이름 (대소문자 무시)::

    nasdaq_full@D  / nasdaq_all@D   NASDAQ 상장 주식 (ETF 제외)
    nyse_full@D                     NYSE (+ NYSE MKT/ARCA 제외) 주식
    us_all@D                        NASDAQ + NYSE 주식

``@D`` 가 없는 이름은 이 어댑터 담당이 아니다 → ValueError (Composite 가
다음 어댑터로 넘긴다).
"""

from __future__ import annotations

import csv
import io
import os
import time
from datetime import date, datetime, timezone
from typing import Any

import httpx
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from data.domain.ports import UniverseProviderPort

_EXCHANGE_FILTERS: dict[str, tuple[str, ...]] = {
    "nasdaq_full": ("NASDAQ",),
    "nasdaq_all": ("NASDAQ",),
    "nyse_full": ("NYSE",),
    "us_all": ("NASDAQ", "NYSE"),
}


def parse_universe_at(name: str) -> tuple[str, date] | None:
    """``"nasdaq_full@2020-01-02"`` → ``("nasdaq_full", date)``. 형식 아니면 None."""
    if "@" not in name:
        return None
    base, _, d = name.strip().lower().partition("@")
    try:
        return base, date.fromisoformat(d)
    except ValueError:
        return None


class AlphaVantageListingUniverseAdapter(UniverseProviderPort):
    """기준일 시점 상장 종목 리스트 (Alpha Vantage LISTING_STATUS, Mongo 캐시)."""

    BASE_URL = "https://www.alphavantage.co/query"
    MIN_DATE = date(2010, 1, 1)

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        mongo_url: str | None = None,
        mongo_db: str | None = None,
        *,
        client: MongoClient | None = None,
        rate_limit_retries: int = 3,
        rate_limit_wait_s: float = 15.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "AlphaVantageListingUniverseAdapter requires the "
                "ALPHAVANTAGE_API_KEY env var or an explicit api_key."
            )
        self._http = http_client or httpx.Client(timeout=60.0)
        url = mongo_url or os.environ.get("MONGO_URL", "mongodb://mongo:27017")
        db_name = mongo_db or os.environ.get("MONGO_DB", "pattern_finder")
        self._client = client or MongoClient(url, serverSelectionTimeoutMS=5000)
        self._coll: Collection = self._client[db_name]["listing_status"]
        self._coll.create_index(
            [("source", ASCENDING), ("date", ASCENDING)], unique=True, name="listing_key",
        )
        self._retries = rate_limit_retries
        self._wait_s = rate_limit_wait_s

    # ---- port ---------------------------------------------------------

    def get_tickers(self, universe: str) -> list[str]:
        parsed = parse_universe_at(universe)
        if parsed is None or parsed[0] not in _EXCHANGE_FILTERS:
            raise ValueError(
                f"Unknown universe: {universe!r}. This adapter serves "
                f"{sorted(_EXCHANGE_FILTERS)} with '@YYYY-MM-DD'."
            )
        base, as_of = parsed
        rows = self.get_listing(as_of)
        wanted = _EXCHANGE_FILTERS[base]
        out: list[str] = []
        for r in rows:
            if r.get("assetType") != "Stock":
                continue
            exch = (r.get("exchange") or "").upper()
            # "NYSE" 는 정확히 NYSE 만 (NYSE MKT / NYSE ARCA 제외)
            if exch not in wanted:
                continue
            sym = (r.get("symbol") or "").strip()
            if not sym or "-" in sym and sym.endswith(("-W", "-R", "-U")):
                continue  # 워런트·권리·유닛
            out.append(sym.replace(".", "-"))  # BRK.B → BRK-B (yfinance 표기)
        return list(dict.fromkeys(out))

    # ---- listing (rows) -----------------------------------------------

    def get_listing(self, as_of: date) -> list[dict[str, Any]]:
        """그날 활성 종목 전체 행 (거래소 무관). Mongo 히트면 네트워크 없음."""
        if as_of < self.MIN_DATE:
            raise ValueError(f"LISTING_STATUS supports dates ≥ {self.MIN_DATE}; got {as_of}")
        key = {"source": "alphavantage", "date": as_of.isoformat()}
        doc = self._coll.find_one(key, {"rows": 1})
        if doc and doc.get("rows"):
            return doc["rows"]
        rows = self._fetch_rows(as_of)
        if rows:
            self._coll.update_one(
                key,
                {"$set": {**key, "rows": rows,
                          "fetched_at": datetime.now(timezone.utc)}},
                upsert=True,
            )
        return rows

    def _fetch_rows(self, as_of: date) -> list[dict[str, Any]]:
        text = None
        for attempt in range(self._retries + 1):
            resp = self._http.get(self.BASE_URL, params={
                "function": "LISTING_STATUS", "date": as_of.isoformat(),
                "state": "active", "apikey": self._api_key,
            })
            if resp.status_code != 200:
                raise ValueError(f"LISTING_STATUS HTTP {resp.status_code}: {resp.text[:200]}")
            text = resp.text
            # 레이트리밋/오류는 JSON 한 줄로 온다 (CSV 헤더 없음)
            if text.lstrip().startswith("{"):
                if attempt < self._retries:
                    time.sleep(self._wait_s)
                    continue
                raise ValueError(f"LISTING_STATUS rate-limited/error: {text[:200]}")
            break
        reader = csv.DictReader(io.StringIO(text or ""))
        rows = []
        for r in reader:
            if not r.get("symbol"):
                continue
            rows.append({
                "symbol": r.get("symbol"), "name": r.get("name"),
                "exchange": r.get("exchange"), "assetType": r.get("assetType"),
                "ipoDate": r.get("ipoDate") or None,
                "delistingDate": (r.get("delistingDate") or None)
                if r.get("delistingDate") not in ("null", "") else None,
            })
        return rows
