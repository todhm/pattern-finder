"""Alpha Vantage 분봉을 (종목, 월) 단위로 받아 parquet 로 캐시하는 배치 로더.

``TIME_SERIES_INTRADAY`` 는 월별 1콜(``month=YYYY-MM``)이라 콜 수 = (종목, 월) 쌍 수. 프리미엄 키 기준 분당 75콜이므로
호출 사이에 ``pace`` 초를 쉬어 한도 응답(재시도 15초 대기)을 피한다. 캐시 파일은 ``{cache_dir}/{SYM}_{interval}_{YYYY-MM}.parquet``,
빈 달(상장 전 등)은 빈 parquet 로 저장해 다시 묻지 않는다. 정규장(09:30~15:59)만 남기고 tz 는 America/New_York.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
from typing import Callable

import pandas as pd

from data.adapters.alphavantage_adapter import AlphaVantageAdapter

RTH = ("09:30", "16:00")


def _month_bounds(month: str) -> tuple[date, date]:
    y, m = int(month[:4]), int(month[5:7])
    first = date(y, m, 1)
    last = date(y + (m == 12), (m % 12) + 1, 1) - pd.Timedelta(days=1)
    return first, last.date() if hasattr(last, "date") else last


def fetch_months(pairs: list[tuple[str, str]], interval: str = "5m", cache_dir: str = "/app/sweep_results/av_intraday_cache",
                 pace: float = 0.85, adapter: AlphaVantageAdapter | None = None,
                 progress: Callable[[int, int, int], None] | None = None) -> dict[str, dict[date, pd.DataFrame]]:
    """pairs = [(symbol, 'YYYY-MM'), ...] → {symbol: {date: 정규장 분봉}}. progress(done, total, api_calls)."""
    adapter = adapter or AlphaVantageAdapter()
    cdir = Path(cache_dir); cdir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[date, pd.DataFrame]] = {}
    calls = 0
    for n, (sym, month) in enumerate(pairs):
        f = cdir / f"{sym}_{interval}_{month}.parquet"
        if f.exists():
            df = pd.read_parquet(f)
        else:
            first, last = _month_bounds(month)
            try:
                df = adapter.fetch_ohlcv(sym, first, last, interval=interval)
            except Exception as e:  # noqa: BLE001 — 한 달 실패는 빈 달로 기록하지 않고 건너뛴다
                print(f"  AV {sym} {month}: {type(e).__name__} {str(e)[:80]}", flush=True)
                df = None
            calls += 1
            time.sleep(pace)
            if df is None:
                continue
            if len(df):
                tm = df.index.strftime("%H:%M")
                df = df[(tm >= RTH[0]) & (tm < RTH[1])]
            df.to_parquet(f)
        if len(df):
            idx = df.index
            if getattr(idx, "tz", None) is None:
                df.index = idx.tz_localize("America/New_York")
            for d, g in df.groupby(df.index.date):
                out.setdefault(sym, {})[d] = g
        if progress and (n + 1) % 25 == 0:
            progress(n + 1, len(pairs), calls)
    return out
