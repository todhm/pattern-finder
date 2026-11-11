"""성장 리스트 종목의 당시 뉴스(Alpha Vantage NEWS_SENTIMENT, 2021-03~) → 일별 뉴스 특징 parquet (설계서 §10).

    docker compose exec backtester python growth_news_cli.py '{"universe": "sp500+nasdaq100", "start": "2021-03-01", "end": "2026-10-24",
        "out": "sweep_results/growth_news_features_2026-10-31.parquet", "cache_dir": "sweep_results/av_news_cache", "pace": 0.85}'

- 리스트(미너비니 성장, 추세 필터 없음)에 한 번이라도 드는 종목만, 반년 창(그 창 또는 다음 창에 리스트에 있을 때) 단위로 받는다. 1000 건이면 창을 반으로 나눠 다시 받는다.
- 기사 날짜 d 의 특징은 **d 이전(≤ d−1) 기사만** 으로 계산(발표 시각 모호성 → 하루 지연, 보수적).
  nsent30 = 30일 관련도 가중 감성 평균 · nvol30 = 30일 기사 수 · nvolrel = nvol30 ÷ (180일 기사 수/6) · nbull30 = Bullish/Somewhat-Bullish 비율 · nneg30 = 부정 키워드 헤드라인 수
- 출력 long parquet: date, ticker, nsent30, nvol30, nvolrel, nbull30, nneg30 (리스트 밖 날도 받은 종목은 전부 계산).
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from data.adapters.alphavantage_news import AlphaVantageNewsAdapter
from data.adapters.wikipedia_universe import default_universe_provider
from data.adapters.yfinance_bulk import download_many
from growth_dip_scalp_cli import load_snapshots, log
from strategy.adapters.growth_dip_scalp import ScalpRules, precompute, weekly_lists
from strategy.adapters.minervini_screener import evaluate_growth

NEG = re.compile(r"cut(s|ting)? (its )?(guidance|outlook|forecast)|lower(s|ed|ing)? (its )?(guidance|outlook|forecast|full-year)|miss(es|ed)? (estimates|expectations|on)|"
                 r"downgrade|investigat|lawsuit|class action|probe|subpoena|recall|fraud|short[- ]seller|plunge|tumble|sinks|slump|warns|disappoint|slashes|"
                 r"layoff|delist|bankrupt|restat", re.I)


def windows(start: date, end: date) -> list[tuple[date, date]]:
    out = []; s = date(start.year, 1 if start.month <= 6 else 7, 1)
    while s <= end:
        e = date(s.year, 6, 30) if s.month == 1 else date(s.year, 12, 31)
        out.append((s, min(e, end))); s = e + timedelta(days=1)
    return out


def fetch_window(ad: AlphaVantageNewsAdapter, sym: str, s: date, e: date, pace: float, stats: dict) -> list[dict]:
    feed = ad.fetch_feed(sym, s, e); stats["calls"] += 1; time.sleep(pace)
    if len(feed) >= 1000 and (e - s).days > 20:                       # 잘림 → 반으로
        mid = s + (e - s) / 2
        return fetch_window(ad, sym, s, mid, pace, stats) + fetch_window(ad, sym, mid + timedelta(days=1), e, pace, stats)
    return feed


def features(sym: str, feed: list[dict], days: pd.DatetimeIndex) -> pd.DataFrame | None:
    rows = []
    for a in feed:
        try:
            d = datetime.strptime(a["time_published"][:8], "%Y%m%d").date()
        except (KeyError, ValueError):
            continue
        ts = next((x for x in a.get("ticker_sentiment", []) if x.get("ticker") == sym), None)
        rel = float(ts["relevance_score"]) if ts and ts.get("relevance_score") else 0.0
        sent = float(ts["ticker_sentiment_score"]) if ts and ts.get("ticker_sentiment_score") else np.nan
        bull = 1.0 if ts and str(ts.get("ticker_sentiment_label", "")).endswith("Bullish") else 0.0
        rows.append((pd.Timestamp(d), rel, rel * sent if not np.isnan(sent) else 0.0, 0.0 if np.isnan(sent) else rel, bull, 1.0 if NEG.search(a.get("title", "")) else 0.0))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["d", "rel", "relsent", "relw", "bull", "neg"]).groupby("d").agg(n=("rel", "size"), relsent=("relsent", "sum"), relw=("relw", "sum"), bull=("bull", "sum"), neg=("neg", "sum"))
    cal = pd.date_range(df.index.min() - pd.Timedelta(days=200), days[-1], freq="D")
    df = df.reindex(cal).fillna(0.0)
    r30 = df.rolling("30D").sum(); r180 = df.rolling("180D").sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = pd.DataFrame({"nsent30": r30.relsent / r30.relw.replace(0, np.nan), "nvol30": r30.n, "nvolrel": r30.n / (r180.n / 6).replace(0, np.nan),
                            "nbull30": r30.bull / r30.n.replace(0, np.nan), "nneg30": r30.neg}, index=cal)
    out = out.shift(1).reindex(days)                                  # d 의 값 = d−1 까지의 30일
    out["ticker"] = sym
    return out.reset_index().rename(columns={"index": "date"})


def main(cfg: dict) -> None:
    prov = default_universe_provider()
    uni = cfg.get("universe", "sp500+nasdaq100")
    tickers = list(dict.fromkeys(t for u in uni.split("+") for t in prov.get_tickers(u.strip()))) if "+" in uni else prov.get_tickers(uni)
    start, end = pd.Timestamp(cfg.get("start", "2021-03-01")), pd.Timestamp(cfg.get("end", date.today().isoformat()))
    t0 = time.time()
    prices = download_many(tickers + ["QQQ"], (start - pd.Timedelta(days=420)).date(), end.date())
    qqq = prices.pop("QQQ"); cal = pd.to_datetime(qqq.index)
    data = precompute({t: d for t, d in prices.items() if t in tickers}, cal)
    log(f"universe {len(data)} · {start.date()} ~ {end.date()} · 가격 {time.time() - t0:.0f}s")
    snaps, _ = load_snapshots(list(data))
    gcache: dict = {}

    def growth_fn(t: str, as_of: date) -> bool:
        k = (t, as_of)
        if k not in gcache:
            g = evaluate_growth(snaps.get(t, []), as_of, eps_min_yoy=0.25, rev_min_yoy=0.25, n_required=3)
            gcache[k] = bool(g.data_available and g.passed)
        return gcache[k]
    lists = weekly_lists(list(data), cal, growth_fn, data, ScalpRules(trend_filter=None, max_vol_ratio=None, min_days_to_earnings=0), start, end)
    wins = windows(start.date(), end.date())
    need: dict[str, set[int]] = {}
    for wk, names in lists.items():
        wi = next(i for i, (s, e) in enumerate(wins) if s <= wk.date() <= e)
        for t in names:
            need.setdefault(t, set()).update({wi, max(0, wi - 1)})   # 그 창 + 앞 창(진입 전 30일 특징)
    n_calls = sum(len(v) for v in need.values())
    log(f"리스트 종목 {len(need)} · (종목,반년창) {n_calls} 쌍 — 캐시 미스는 1콜 이상 (분당 ≈{60 / float(cfg.get('pace', 0.85)):.0f})")
    ad = AlphaVantageNewsAdapter(cache_dir=cfg.get("cache_dir", "sweep_results/av_news_cache"))
    pace = float(cfg.get("pace", 0.85)); stats = {"calls": 0}
    days = cal[(cal >= start) & (cal <= end)]
    frames = []; t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    workers = int(cfg.get("workers", 3))                                 # 콜당 지연 ≈1.6s + pace 라 3 워커 ≈ 분당 60 (한도 75)

    def one(item):
        t, wis = item
        feed: list[dict] = []
        for wi in sorted(wis):
            s, e = wins[wi]
            feed += fetch_window(ad, t, s, e, pace, stats)
        return t, feed
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (t, feed) in enumerate(ex.map(one, sorted(need.items()))):
            f = features(t, feed, days)
            if f is not None:
                frames.append(f)
            if (i + 1) % 25 == 0:
                log(f"  뉴스 {i + 1}/{len(need)} (콜 {stats['calls']}, {time.time() - t0:.0f}s) {t}: 기사 {len(feed)}")
    out = pd.concat(frames, ignore_index=True).rename(columns={"Date": "date"})
    out = out[["date", "ticker", "nsent30", "nvol30", "nvolrel", "nbull30", "nneg30"]]
    out.to_parquet(cfg["out"])
    log(f"저장 {cfg['out']}: 행 {len(out)}, 종목 {out.ticker.nunique()}, 콜 {stats['calls']}, {time.time() - t0:.0f}s")
    log("특징 요약: " + out[["nsent30", "nvol30", "nvolrel", "nbull30", "nneg30"]].describe().round(3).to_string().replace("\n", " | "))


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
