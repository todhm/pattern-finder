"""급등주 탐색 — 종목별 일봉 원본 수집 (Alpha Vantage TIME_SERIES_DAILY_ADJUSTED, 종목당 1콜, 전체 히스토리).

alphafolio `us_daily`(2017-11~) 는 ① 현재 상장 종목 위주(폐지 9,317 중 가격 383) 라 생존 편향이 있고 ② 분할 미조정 원가격이라
역분할이 가짜 급등으로 잡힌다. 그래서 `us_listing_status` 의 종목(폐지·상장 모두) 을 AV 에서 다시 받아
**원가격(o,h,l,c,v) + 조정 종가(adj) + 배당(div) + 분할 계수(split)** 를 그대로 연구용 parquet 로 저장한다(DB 는 읽기만).
실행: docker compose exec backtester python surge_delisted_cli.py '{"list": "/app/sweep_results/surge_delisted_list.txt", "cache": "/app/sweep_results/surge_av_cache", "workers": 2, "pace": 1.0}'
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

COLS = {"1. open": "o", "2. high": "h", "3. low": "l", "4. close": "c", "5. adjusted close": "adj", "6. volume": "v", "7. dividend amount": "div", "8. split coefficient": "split"}


def main(cfg: dict) -> None:
    from data.adapters.alphavantage_adapter import AlphaVantageAdapter, _DAILY_KEY
    cache = Path(cfg.get("cache", "/app/sweep_results/surge_av_cache")); cache.mkdir(parents=True, exist_ok=True)
    syms = [ln.split(",")[0] for ln in Path(cfg["list"]).read_text().splitlines() if ln.strip()]
    fn = lambda s: cache / (s.replace("/", "_") + ".parquet")  # noqa: E731
    todo = [s for s in syms if not fn(s).exists()]
    print(f"종목 {len(syms)} · 받을 것 {len(todo)}", file=sys.stderr, flush=True)
    pace = float(cfg.get("pace", 1.0)); workers = int(cfg.get("workers", 2)); lock = threading.Lock(); n = {"ok": 0, "empty": 0, "i": 0}
    local = threading.local(); since = pd.Timestamp(cfg.get("since", "2015-01-01"))

    def one(s: str) -> None:
        if not hasattr(local, "a"):
            local.a = AlphaVantageAdapter()
        t0 = time.time(); ok = False; df = pd.DataFrame(columns=list(COLS.values()))
        try:
            payload = local.a._query({"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": s, "outputsize": "full", "apikey": local.a._api_key},
                                     series_key=_DAILY_KEY, label=f"{s} 1d", any_note_is_limit=True)
            series = payload.get(_DAILY_KEY)
            if series:
                raw = pd.DataFrame.from_dict(series, orient="index").astype(float).rename(columns=COLS)
                raw.index = pd.DatetimeIndex(pd.to_datetime(raw.index)); df = raw.sort_index()[list(COLS.values())]; df = df[df.index >= since]; ok = len(df) > 0
        except Exception as e:  # noqa: BLE001
            if "rate limit" in str(e).lower():
                time.sleep(20); return                      # 파일을 안 남겨 다음 실행에서 다시 받는다
        df.to_parquet(fn(s))
        with lock:
            n["i"] += 1; n["ok" if ok else "empty"] += 1
            if n["i"] % 200 == 0 or n["i"] == len(todo):
                print(f"[{n['i']}/{len(todo)}] ok {n['ok']} · 없음 {n['empty']}", file=sys.stderr, flush=True)
        time.sleep(max(0.0, pace * workers - (time.time() - t0)))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {})
