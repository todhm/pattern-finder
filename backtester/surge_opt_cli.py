"""옵션 탐색(O 계열) 자료 — Alpha Vantage HISTORICAL_OPTIONS 일별 체인(실제 매수·매도 호가)을 날짜별 parquet 로 캐시.

 fetch  '{"symbols": ["TQQQ", "QQQ"], "start": "2010-02-11", "end": "2023-12-29"}'
        → /app/sweep_results/surge_opt_cache/{SYM}/{YYYY-MM-DD}.parquet (자료 없는 날은 빈 파일 = 재조회 안 함)
봉인 구간(2024-01~)은 후보가 생겨 장부에 열람을 적은 뒤에만 end 를 늘려 받는다.
실행: docker compose exec backtester python surge_opt_cli.py fetch '{...}'
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

SR = Path("/app/sweep_results")
CACHE = SR / "surge_opt_cache"
NUM = ["strike", "bid", "ask", "mark", "last", "volume", "open_interest", "implied_volatility", "delta"]
_lock = threading.Lock(); _last = [0.0]


def _pace(gap: float) -> None:
    with _lock:
        wait = _last[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def fetch_day(sym: str, day: str, key: str, gap: float) -> int:
    path = CACHE / sym / f"{day}.parquet"
    if path.exists():
        return -1
    for attempt in range(4):
        _pace(gap)
        try:
            j = requests.get("https://www.alphavantage.co/query", params={"function": "HISTORICAL_OPTIONS", "symbol": sym, "date": day, "apikey": key}, timeout=120).json()
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1)); continue
        if "data" not in j:                                              # 분당 한도 안내 등 → 쉬고 재시도
            time.sleep(15 * (attempt + 1)); continue
        df = pd.DataFrame(j["data"])
        if df.empty:
            pd.DataFrame(columns=["contractID", "expiration", "type", *NUM]).to_parquet(path); return 0
        for c in NUM:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32" if c != "strike" else "float64")
        df[["contractID", "expiration", "type", *NUM]].to_parquet(path, compression="zstd"); return len(df)
    return -2


def fetch(cfg: dict) -> None:
    key = os.environ["ALPHAVANTAGE_API_KEY"]; gap = float(cfg.get("gap", 0.85))
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(cfg.get("start", "2010-02-11"), cfg.get("end", "2023-12-29"))]
    for sym in cfg.get("symbols", ["TQQQ", "QQQ"]):
        (CACHE / sym).mkdir(parents=True, exist_ok=True); done = [0, 0, 0]

        def one(d, sym=sym, done=done):
            n = fetch_day(sym, d, key, gap); done[0] += 1; done[1] += n > 0; done[2] += n == -2
            if done[0] % 200 == 0:
                print(f"{sym} {done[0]}/{len(days)} 자료 있는 날 {done[1]} 실패 {done[2]} ({d})", file=sys.stderr, flush=True)

        with ThreadPoolExecutor(int(cfg.get("workers", 3))) as ex:
            list(ex.map(one, days))
        print(f"{sym} 끝: {done[0]} 일 중 새로 받은 자료 {done[1]}, 실패 {done[2]}", flush=True)


if __name__ == "__main__":
    {"fetch": fetch}[sys.argv[1]](json.loads(sys.argv[2]) if len(sys.argv) > 2 else {})
