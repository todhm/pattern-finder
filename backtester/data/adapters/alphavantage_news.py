"""Alpha Vantage ``NEWS_SENTIMENT`` 뉴스 어댑터 (NewsCatalystPort 구현).

EODHD 뉴스 키가 없을 때의 대체 소스. 기간(``time_from``/``time_to``)을
지정해 point-in-time 헤드라인을 받는다 — 미너비니 2-8(가이던스 하향
없음) 판정에 쓴다. 커버리지는 대략 2022년 이후가 두텁고 그 이전은
얇다 (빈 결과 = '하향 증거 없음'이지 '확인됨'이 아님).

응답은 ``(symbol, start, end)`` 키로 디스크 캐시 (과거 구간은 불변).
포트 계약대로 실패 시 raise 대신 ``[]``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

import httpx

from data.domain.models import NewsEvent
from data.domain.ports import NewsCatalystPort

log = logging.getLogger(__name__)


class AlphaVantageNewsAdapter(NewsCatalystPort):
    DEFAULT_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str | None = None,
        http_client: httpx.Client | None = None,
        cache_dir: str | os.PathLike = "/tmp/pattern-finder-cache/news-av",
        rate_retry_sleep: float = 6.0,
        max_rate_retries: int = 5,
    ) -> None:
        key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not key:
            raise ValueError("ALPHAVANTAGE_API_KEY is not set — AlphaVantageNewsAdapter")
        self._api_key = key
        self._client = http_client or httpx.Client(timeout=60.0)
        self._cache_dir = Path(cache_dir)
        self._sleep = rate_retry_sleep
        self._retries = max_rate_retries

    def fetch_news(
        self,
        symbol: str,
        start: date,
        end: date,
        limit_per_day: int | None = None,
    ) -> list[NewsEvent]:
        feed = self._cached_feed(symbol, start, end)
        out: list[NewsEvent] = []
        for row in feed:
            try:
                published = datetime.strptime(row["time_published"][:15], "%Y%m%dT%H%M%S")
            except (KeyError, ValueError):
                continue
            if not (start <= published.date() <= end):
                continue
            score = row.get("overall_sentiment_score")
            out.append(NewsEvent(
                symbol=symbol, published_at=published, title=row.get("title", ""),
                sentiment=float(score) if score is not None else None,
                source="alphavantage",
            ))
        out.sort(key=lambda n: n.published_at)
        if limit_per_day is not None:
            kept: dict[date, int] = {}
            capped = []
            for n in out:
                if kept.get(n.date, 0) < limit_per_day:
                    kept[n.date] = kept.get(n.date, 0) + 1
                    capped.append(n)
            out = capped
        return out

    # ---- internals ----------------------------------------------------

    def _cached_feed(self, symbol: str, start: date, end: date) -> list[dict]:
        key = hashlib.sha1(f"{symbol}|{start}|{end}".encode()).hexdigest()[:16]
        path = self._cache_dir / f"{symbol}_{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except ValueError:
                pass
        feed = self._query(symbol, start, end)
        if feed is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(feed))
        return feed or []

    def _query(self, symbol: str, start: date, end: date) -> list[dict] | None:
        params = {
            "function": "NEWS_SENTIMENT", "tickers": symbol,
            "time_from": start.strftime("%Y%m%dT0000"),
            "time_to": end.strftime("%Y%m%dT2359"),
            "limit": 1000, "sort": "EARLIEST", "apikey": self._api_key,
        }
        for _ in range(self._retries + 1):
            try:
                payload = self._client.get(self.DEFAULT_BASE_URL, params=params).json()
            except Exception as exc:  # noqa: BLE001 — 포트 계약: 실패 시 []
                log.warning("AV news %s: %s", symbol, exc)
                return None
            if "feed" in payload:
                return payload["feed"]
            note = str(payload.get("Information") or payload.get("Note") or "")
            if note and "No articles found" not in note and "Invalid inputs" not in note:
                time.sleep(self._sleep)  # 분당 한도 — 잠깐 쉬고 재시도
                continue
            return []  # 기사 없음 / 미지원 티커 — 캐시해도 되는 '빈 결과'
        return None  # 한도 소진 — 캐시하지 않음
