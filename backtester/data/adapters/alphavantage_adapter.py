"""Alpha Vantage market-data adapter (sub-daily 전용).

REST endpoint: ``GET https://www.alphavantage.co/query`` with
``function=TIME_SERIES_INTRADAY``. Auth via ``apikey`` query param;
key loaded from the ``ALPHAVANTAGE_API_KEY`` env var
(``secrets/alphafolio.env`` — alphafolio_data 파이프라인과 공유).

왜 이 소스가 인트라데이 체인의 **최우선**인가 (2026-08 검증):

- ``month=YYYY-MM`` 파라미터로 2000년대까지 월 단위 히스토리 조회
  가능 — EODHD(키 만료 401), Polygon(429), yfinance(최근 ~60일)이
  못 주는 **장기 15분봉**을 유일하게 공급.
- 응답 타임스탬프는 US/Eastern (API 문서 명시). yfinance 관례에
  맞춰 ``America/New_York`` tz-aware 인덱스로 반환 — 라우팅/캐시/
  RTH 필터가 소스에 무관하게 동일한 인덱스를 본다.
- 확장시간(pre/post) 봉 포함 — 정규장 필터는 downstream
  (:class:`RegularSessionFilterAdapter`)의 몫.

월별 1콜이므로 긴 구간은 콜 수 = 개월 수. 분당 한도 응답
(JSON "Note"/"Information")은 짧게 대기 후 재시도하고, 재시도
소진 시 raise — 상위 :class:`FallbackMarketDataAdapter`가 다음
소스로 라우팅한다. 일봉(1d+)은 지원하지 않는다 (라우팅 레이어가
일봉을 이 체인으로 보내지 않음).
"""

from __future__ import annotations

import os
import time
from datetime import date

import httpx
import pandas as pd

from data.domain.ports import MarketDataPort

NY_TZ = "America/New_York"

_INTRADAY_INTERVALS: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "1h": "60min",
}

_RATE_LIMIT_MARKERS = ("rate limit", "calls per minute", "premium")


class AlphaVantageAdapter(MarketDataPort):
    """월 단위 TIME_SERIES_INTRADAY 페치 구현."""

    DEFAULT_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        rate_retry_sleep: float = 15.0,
        max_rate_retries: int = 4,
    ) -> None:
        key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
        if not key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY is not set — AlphaVantageAdapter "
                "requires an API key (secrets/alphafolio.env)."
            )
        self._api_key = key
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._client = http_client or httpx.Client(timeout=60.0)
        self._rate_retry_sleep = rate_retry_sleep
        self._max_rate_retries = max_rate_retries

    def fetch_ohlcv(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval not in _INTRADAY_INTERVALS:
            raise ValueError(
                f"AlphaVantageAdapter supports only sub-daily intervals "
                f"{sorted(_INTRADAY_INTERVALS)} (got {interval!r})."
            )
        av_interval = _INTRADAY_INTERVALS[interval]

        frames: list[pd.DataFrame] = []
        for month in _iter_months(start, end):
            payload = self._fetch_month(symbol, av_interval, month)
            series = payload.get(f"Time Series ({av_interval})")
            if series:
                frames.append(self._to_frame(series))

        if not frames:
            return _empty_frame()
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        lo = pd.Timestamp(start, tz=NY_TZ)
        hi = pd.Timestamp(end, tz=NY_TZ) + pd.Timedelta(days=1)
        return df[(df.index >= lo) & (df.index < hi)]

    # ---- internals -------------------------------------------------

    def _fetch_month(self, symbol: str, av_interval: str, month: str) -> dict:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": av_interval,
            "month": month,
            "outputsize": "full",
            "apikey": self._api_key,
        }
        for attempt in range(self._max_rate_retries + 1):
            resp = self._client.get(self._base_url, params=params)
            if resp.status_code != 200:
                raise ValueError(
                    f"Alpha Vantage {symbol} {av_interval} {month}: "
                    f"HTTP {resp.status_code} {resp.text[:200]}"
                )
            payload = resp.json()
            if f"Time Series ({av_interval})" in payload:
                return payload
            if "Error Message" in payload:
                raise ValueError(
                    f"Alpha Vantage {symbol} {month}: {payload['Error Message'][:200]}"
                )
            note = str(
                payload.get("Note") or payload.get("Information") or ""
            )
            if any(m in note.lower() for m in _RATE_LIMIT_MARKERS):
                if attempt < self._max_rate_retries:
                    time.sleep(self._rate_retry_sleep)
                    continue
                raise RuntimeError(
                    f"Alpha Vantage rate limit for {symbol} {month}: {note[:200]}"
                )
            # 데이터 없는 달 (상장 이전 등) — 마커 없이 빈 응답.
            return payload
        raise RuntimeError(f"Alpha Vantage {symbol} {month}: retries exhausted")

    @staticmethod
    def _to_frame(series: dict) -> pd.DataFrame:
        idx = pd.DatetimeIndex(pd.to_datetime(list(series.keys())))
        df = pd.DataFrame(
            {
                "Open": [float(v["1. open"]) for v in series.values()],
                "High": [float(v["2. high"]) for v in series.values()],
                "Low": [float(v["3. low"]) for v in series.values()],
                "Close": [float(v["4. close"]) for v in series.values()],
                "Volume": [float(v["5. volume"]) for v in series.values()],
            },
            index=idx,
        )
        # AV 인트라데이 타임스탬프는 US/Eastern (문서 명시). DST 전환의
        # 모호/결측 시각은 드랍 — 장중 봉엔 실질 영향 없음.
        df.index = df.index.tz_localize(
            NY_TZ, ambiguous="NaT", nonexistent="NaT"
        )
        return df[df.index.notna()].sort_index()


def _iter_months(start: date, end: date):
    """start~end를 덮는 YYYY-MM 문자열 나열."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex([], tz=NY_TZ),
    )
