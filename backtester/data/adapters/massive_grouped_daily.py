"""Massive(Polygon) grouped-daily — 날짜 하나로 미국 전 종목 일봉을 받는 어댑터.

``GET /v2/aggs/grouped/locale/us/market/stocks/{date}?adjusted=true`` 는 1콜에
약 12,000 종목의 OHLCV 를 돌려준다 (2026-10 실측 2.4초). 종목별 히스토리
API(Alpha Vantage 분당 75콜)로 캐시 앞단을 채우면 5,000 종목에 1시간이 걸리던
것을 날짜당 1콜로 대체한다.

분할은 ``GET /v3/reference/splits?execution_date.gte=..&execution_date.lte=..``
로 구간 한 번에 받는다 — 분할 전에 캐시된 종목은 과거 문서가 분할 전 가격이라
캐시를 비우고 종목별 소스(조정가)로 다시 받게 해야 한다.

심볼 표기: Polygon 은 ``BRK.B``, Alpha Vantage LISTING_STATUS 유니버스는
``BRK-B``. 반환 index 는 ``.`` → ``-`` 로 통일한다.
"""

from __future__ import annotations

import os
import time
from datetime import date

import httpx
import pandas as pd

from data.domain.ports import BulkDailyBarsPort

_COLS = ["Open", "High", "Low", "Close", "Volume"]


class MassiveGroupedDailyAdapter(BulkDailyBarsPort):
    DEFAULT_BASE_URL = "https://api.polygon.io"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        rate_retry_sleep: float = 12.0,
        max_rate_retries: int = 5,
    ) -> None:
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "MassiveGroupedDailyAdapter requires the MASSIVE_API_KEY env var "
                "or an explicit api_key argument."
            )
        self._base_url = (
            base_url or os.environ.get("MASSIVE_API_BASE") or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)
        self._rate_retry_sleep = rate_retry_sleep
        self._max_rate_retries = max_rate_retries

    # ---- BulkDailyBarsPort ------------------------------------------

    def fetch_day(self, day: date) -> pd.DataFrame:
        url = f"{self._base_url}/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}"
        payload = self._get(url, {"adjusted": "true"})
        if payload.get("status") not in ("OK", "DELAYED"):
            raise ValueError(
                f"Massive grouped daily {day}: status={payload.get('status')} "
                f"{str(payload.get('error') or payload.get('message') or '')[:200]}"
            )
        rows = payload.get("results") or []
        if not rows:
            return pd.DataFrame(columns=_COLS)
        df = pd.DataFrame.from_records(rows)
        df = df.rename(columns={"T": "symbol", "o": "Open", "h": "High", "l": "Low",
                                "c": "Close", "v": "Volume"})
        df = df.dropna(subset=["symbol"])
        df["symbol"] = df["symbol"].astype(str).str.replace(".", "-", regex=False)
        df = df.drop_duplicates(subset=["symbol"], keep="first").set_index("symbol")
        for c in _COLS:
            if c not in df.columns:
                df[c] = float("nan")
        return df[_COLS].astype(float)

    def split_symbols(self, start: date, end: date) -> list[str]:
        url = f"{self._base_url}/v3/reference/splits"
        params = {"execution_date.gte": start.isoformat(),
                  "execution_date.lte": end.isoformat(), "limit": 1000}
        out: list[str] = []
        seen: set[str] = set()
        next_url: str | None = url
        first = True
        while next_url:
            payload = self._get(next_url, params if first else {})
            first = False
            for r in payload.get("results") or []:
                t = str(r.get("ticker") or "").replace(".", "-")
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
            next_url = payload.get("next_url")
        return out

    # ---- internals --------------------------------------------------

    def _get(self, url: str, params: dict) -> dict:
        q = dict(params)
        q["apiKey"] = self._api_key
        for attempt in range(self._max_rate_retries + 1):
            resp = self._client.get(url, params=q)
            if resp.status_code == 429 and attempt < self._max_rate_retries:
                time.sleep(self._rate_retry_sleep)
                continue
            if resp.status_code != 200:
                raise ValueError(
                    f"Massive {url.split('?')[0].rsplit('/', 1)[-1]}: "
                    f"HTTP {resp.status_code} {resp.text[:200]}"
                )
            return resp.json()
        raise RuntimeError(f"Massive rate limit: retries exhausted for {url}")
