"""FRED (St. Louis Fed) 경제지표 어댑터 — 무키 CSV 엔드포인트.

``fredgraph.csv``는 API 키 없이 시리즈 전체를 CSV로 내려준다.
``cosd``/``coed``(chart start/end date)를 명시하지 않으면 일부
시리즈(예: BAMLH0A0HYM2)가 최근 몇 년만 반환하므로 항상 명시한다.

디스크 캐시: 요청 키(series, start, end)별 parquet. 과거 구간은
영구, ``end >= today`` 요청은 TTL(기본 24h) 내에서만 재사용 —
일/주/월 단위로 갱신되는 지표라 장중 신선도가 필요 없다.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.domain.ports import EconomicSeriesPort


class FredCsvAdapter(EconomicSeriesPort):
    BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(
        self,
        cache_dir: str | os.PathLike = "/tmp/pattern-finder-cache",
        ttl_hours: int = 24,
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._ttl = timedelta(hours=ttl_hours)

    def fetch_series(self, series_id: str, start: date, end: date) -> pd.Series:
        path = self._cache_path(series_id, start, end)
        if path.exists():
            fresh = (
                end < date.today()
                or datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
                < self._ttl
            )
            if fresh:
                try:
                    return pd.read_parquet(path)[series_id]
                except Exception:
                    path.unlink(missing_ok=True)

        url = (
            f"{self.BASE_URL}?id={series_id}"
            f"&cosd={start.isoformat()}&coed={end.isoformat()}"
        )
        df = pd.read_csv(url, na_values=".")
        # 첫 컬럼 = 관측일 (FRED가 'DATE' ↔ 'observation_date'로
        # 바꿔온 이력이 있어 위치 기반으로 잡는다), 둘째 = 값.
        date_col, value_col = df.columns[0], df.columns[1]
        series = pd.Series(
            pd.to_numeric(df[value_col], errors="coerce").values,
            index=pd.to_datetime(df[date_col]),
            name=series_id,
        ).dropna()
        if series.empty:
            raise ValueError(f"FRED returned no data for {series_id}")
        try:
            series.to_frame().to_parquet(path)
        except Exception:
            pass
        return series

    def _cache_path(self, series_id: str, start: date, end: date) -> Path:
        key = f"fred_{series_id}_{start.isoformat()}_{end.isoformat()}"
        safe = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self._cache_dir / f"fred_{series_id}_{safe}.parquet"
