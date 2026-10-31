"""yfinance 다중 종목 일괄 일봉 다운로드 — 스크리너용 빠른 경로.

``yf.download([...])`` 는 한 요청에 수백 종목을 받아 종목당 0.05초 수준이다 (단일 ``Ticker.history`` 는
종목당 0.5~1초). Mongo 캐시 스택을 거치지 않으므로 **캐시에 기록되지 않고** 폴백 소스도 없다.
조정 방식은 단일 어댑터(``YFinanceAdapter`` → ``history()`` 기본 auto_adjust=True)와 같다.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import yfinance as yf

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _download_with_retry(batch: list[str], start: date, end: date, retries: int) -> pd.DataFrame | None:
    """yfinance 는 429(크럼 제한)에 걸리면 배치의 절반 이상을 'Failed download' 로 조용히 비운다 →
    Close 가 전부 NaN 인 종목이 배치의 1/3 을 넘으면 잠시 쉬고 그 종목만 다시 받는다."""
    raw = yf.download(batch, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                      group_by="ticker", threads=True, auto_adjust=True, progress=False)
    if raw is None or raw.empty or len(batch) == 1:
        return raw
    for attempt in range(retries):
        lv0 = raw.columns.get_level_values(0)
        missing = [t for t in batch if t not in lv0 or raw[t]["Close"].isna().all()]
        if len(missing) <= len(batch) / 3:
            break
        time.sleep(5 * (attempt + 1))
        again = yf.download(missing, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                            group_by="ticker", threads=True, auto_adjust=True, progress=False)
        if again is None or again.empty:
            continue
        if len(missing) == 1:
            again = pd.concat({missing[0]: again}, axis=1)
        keep = [t for t in lv0.unique() if t not in missing]
        raw = pd.concat([raw[keep], again], axis=1) if keep else again
    return raw


def download_many(tickers: list[str], start: date, end: date, chunk: int = 200,
                  progress: Callable[[int, int], None] | None = None, retries: int = 2) -> dict[str, pd.DataFrame]:
    """{ticker: OHLCV 일봉} — 데이터가 없는 종목은 빠진다. ``end`` 포함. 429 로 비면 ``retries`` 번 재시도."""
    out: dict[str, pd.DataFrame] = {}
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        raw = _download_with_retry(batch, start, end, retries)
        if raw is None or raw.empty:
            continue
        if len(batch) == 1:
            raw = pd.concat({batch[0]: raw}, axis=1)
        for t in batch:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].dropna(subset=["Close"])
            if df.empty:
                continue
            df = df[[c for c in COLS if c in df.columns]].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None) if getattr(df.index, "tz", None) is not None else pd.to_datetime(df.index)
            out[t] = df
        if progress:
            progress(min(i + chunk, len(tickers)), len(tickers))
    return out


def download_intraday(tickers: list[str], interval: str = "5m", days: int = 59, chunk: int = 100,
                      progress: Callable[[int, int], None] | None = None) -> dict[str, pd.DataFrame]:
    """{ticker: 분봉(정규장만, tz America/New_York)} — yfinance 는 5m/15m 최근 60일, 1m 7일만 준다."""
    out: dict[str, pd.DataFrame] = {}
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        raw = yf.download(batch, period=f"{days}d", interval=interval, group_by="ticker", threads=True,
                          auto_adjust=True, prepost=False, progress=False)
        if raw is None or raw.empty:
            continue
        if len(batch) == 1:
            raw = pd.concat({batch[0]: raw}, axis=1)
        for t in batch:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].dropna(subset=["Close"])
            if df.empty:
                continue
            df = df[[c for c in COLS if c in df.columns]].copy()
            idx = pd.to_datetime(df.index)
            idx = idx.tz_convert("America/New_York") if getattr(idx, "tz", None) is not None else idx.tz_localize("America/New_York")
            df.index = idx
            df = df[(df.index.strftime("%H:%M") >= "09:30") & (df.index.strftime("%H:%M") < "16:00")]
            out[t] = df
        if progress:
            progress(min(i + chunk, len(tickers)), len(tickers))
    return out
