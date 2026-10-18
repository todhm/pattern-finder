"""캐시 앞단 채우기 CLI — 스크리너 돌리기 전(또는 크론으로 장 마감 후) 실행.

    docker compose exec backtester python cache_fill_cli.py '{"universe": "nasdaq_full@기준일", "as_of": "2026-10-18"}'
    docker compose exec backtester python cache_fill_cli.py '{"universe": "sp500", "as_of": "2026-10-18", "time_fetch": true}'

옵션: lookback(기본 15 거래일), max_tickers, time_fetch(채운 뒤 종목별 fetch 루프를
실제로 돌려 소요 시간·느린 종목 수를 잰다 — 44 페이지 가격 수집 단계와 같은 창).
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta

from data.adapters.cache_frontier_fill import fill_recent_days
from data.adapters.composed_market_data import build_daily_cache_filler, build_default_market_data
from data.adapters.wikipedia_universe import default_universe_provider


def _tickers(choice: str, as_of: date) -> list[str]:
    provider = default_universe_provider()
    if choice.endswith("@기준일"):
        return provider.get_tickers(f"{choice[:-len('@기준일')]}@{as_of.isoformat()}")
    if choice == "sp500+nasdaq100":
        return list(dict.fromkeys(provider.get_tickers("sp500") + provider.get_tickers("nasdaq100")))
    return provider.get_tickers(choice)


def main(cfg: dict) -> dict:
    as_of = date.fromisoformat(cfg.get("as_of") or date.today().isoformat())
    t0 = time.perf_counter()
    tickers = _tickers(cfg.get("universe", "nasdaq_full@기준일"), as_of)
    if cfg.get("max_tickers"):
        tickers = tickers[: int(cfg["max_tickers"])]
    t_univ = time.perf_counter() - t0
    filler = build_daily_cache_filler()
    if filler is None:
        return {"error": "ALPHAVANTAGE_API_KEY / MASSIVE_API_KEY 필요"}
    cache, bulk, chain = filler
    t0 = time.perf_counter()
    rep = fill_recent_days(cache, bulk, tickers + ["SPY", "QQQ"], as_of,
                           lookback_bdays=int(cfg.get("lookback", 15)), marker_caches=chain,
                           log=lambda m: print(m, file=sys.stderr))
    out = {
        "as_of": as_of.isoformat(), "universe": cfg.get("universe", "nasdaq_full@기준일"),
        "tickers": len(tickers), "universe_sec": round(t_univ, 1),
        "fill_sec": round(time.perf_counter() - t0, 1), "fill": rep.summary(),
        "days_filled": [d.isoformat() for d in rep.days_filled],
        "days_holiday": [d.isoformat() for d in rep.days_holiday],
        "days_covered": [d.isoformat() for d in rep.days_skipped_covered],
        "days_deferred": [d.isoformat() for d in rep.days_deferred],
        "splits_dropped": rep.splits_dropped, "dead_marked": rep.dead_marked, "errors": rep.errors,
    }
    if cfg.get("time_fetch"):
        md = build_default_market_data()
        start = as_of - timedelta(days=460)
        t0 = time.perf_counter()
        slow, failed, n = [], 0, 0
        for i, t in enumerate(tickers):
            t1 = time.perf_counter()
            try:
                md.fetch_ohlcv(t, start, as_of)
                n += 1
            except Exception:  # noqa: BLE001
                failed += 1
            dt = time.perf_counter() - t1
            if dt > 0.3:
                slow.append((t, round(dt, 2)))
            if (i + 1) % 500 == 0:
                print(f"fetch {i + 1}/{len(tickers)} {time.perf_counter() - t0:.0f}s slow={len(slow)}", file=sys.stderr)
        out["fetch_sec"] = round(time.perf_counter() - t0, 1)
        out["fetch_ok"] = n
        out["fetch_failed"] = failed
        out["fetch_slow_count"] = len(slow)
        out["fetch_slow_sample"] = slow[:20]
    return out


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    print(json.dumps(main(cfg), ensure_ascii=False, indent=2))
