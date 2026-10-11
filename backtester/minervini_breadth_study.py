"""0-3 시장 폭(신고가 vs 신저가) 일별 시계열 연구 — 이 조건이 지수 성과를 가르는가.

종목 풀 = 여러 기준일의 ``nasdaq_full@기준일`` 유니버스 합집합. 각 종목 전체 일봉을
한 번 읽어 매일 다음을 집계한다 (그날 상장돼 252봉 이상 있는 종목만 — point-in-time 근사):

    NH = 종가 ≥ 52주 고가 × 0.95,  NL = 종가 ≤ 52주 저가 × 1.05
    전체 / 유동성(종가 ≥ $10 · 20일 평균 거래대금 ≥ $5M) 두 가지

그리고 같은 날 SPY·QQQ 의 0-1(200일선)·0-1b(50일선 20일 기울기) 게이트와 QQQ 이후
21·63거래일 수익률, 63거래일 최대 낙폭을 붙여 "게이트 통과일 중 0-3 실패/통과" 를 비교한다.

    docker compose exec backtester python minervini_breadth_study.py \\
        '{"pool_dates": ["2019-03-01", ...], "start": "2019-01-01", "out": "/tmp/breadth_daily.csv"}'
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import pandas as pd

from data.adapters.alphavantage_adapter import AlphaVantageAdapter
from data.adapters.mongo_day_cache import MongoDayCacheAdapter
from data.adapters.wikipedia_universe import default_universe_provider


def _log(m: str) -> None:
    print(m, file=sys.stderr, flush=True)


def _load(md, t: str, start: date, end: date) -> pd.DataFrame | None:
    try:
        df = md.fetch_ohlcv(t, start, end)
    except Exception:  # noqa: BLE001
        return None
    if df is None or len(df) < 260:
        return None
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


def ticker_flags(df: pd.DataFrame) -> pd.DataFrame:
    """종목 1개의 일별 NH/NL/유동성 플래그 (252봉 미만 구간은 제외)."""
    c = df["Close"]
    hi = df["High"].rolling(252).max()
    lo = df["Low"].rolling(252).min()
    dv = (c * df["Volume"]).rolling(20).mean()
    valid = hi.notna()
    nh = (c >= 0.95 * hi) & valid
    nl = (c <= 1.05 * lo) & valid & ~nh
    liq = (c >= 10) & (dv >= 5e6) & valid
    return pd.DataFrame({"n": valid, "nh": nh, "nl": nl, "n_liq": liq,
                         "nh_liq": nh & liq, "nl_liq": nl & liq}).astype(int)


def gate_series(df: pd.DataFrame) -> pd.DataFrame:
    """지수 1개의 일별 0-1·0-1b (evaluate_market 과 같은 정의, 20일 기울기)."""
    c = df["Close"]
    s200, s50 = c.rolling(200).mean(), c.rolling(50).mean()
    trend = (c > s200) & (s200 > s200.shift(21))
    short = ~((c < s50) & (s50 < s50.shift(20)))
    return pd.DataFrame({"trend": trend, "short": short})


def build(cfg: dict) -> pd.DataFrame:
    md = MongoDayCacheAdapter(AlphaVantageAdapter(), source_name="alphavantage")
    uni = default_universe_provider()
    start = date.fromisoformat(cfg.get("start", "2019-01-01"))
    end = date.fromisoformat(cfg.get("end", date.today().isoformat()))
    pool: set[str] = set()
    for d in cfg["pool_dates"]:
        pool |= set(uni.get_tickers(f"nasdaq_full@{d}"))
    pool = sorted(pool)
    _log(f"종목 풀 {len(pool)}")
    load_start = start - timedelta(days=400)
    totals: pd.DataFrame | None = None
    done = 0
    with ThreadPoolExecutor(max_workers=int(cfg.get("workers", 6))) as ex:
        for df in ex.map(lambda t: _load(md, t, load_start, end), pool):
            done += 1
            if done % 500 == 0:
                _log(f"  {done}/{len(pool)}")
            if df is None:
                continue
            f = ticker_flags(df)
            totals = f if totals is None else totals.add(f, fill_value=0)
    out = totals.loc[pd.Timestamp(start):].copy()
    idx = {}
    for sym in ("SPY", "QQQ"):
        idf = _load(md, sym, load_start, end)
        idx[sym] = idf
        g = gate_series(idf)
        out[f"{sym}_trend"] = g["trend"].reindex(out.index)
        out[f"{sym}_short"] = g["short"].reindex(out.index)
    q = idx["QQQ"]["Close"].reindex(out.index)
    out["qqq_fwd21"] = q.shift(-21) / q - 1
    out["qqq_fwd63"] = q.shift(-63) / q - 1
    fut_min = q[::-1].rolling(63, min_periods=1).min()[::-1].shift(-1)
    out["qqq_dd63"] = fut_min / q - 1
    out = out[out["n"] >= 500]  # 데이터 얇은 날 제외
    return out


def analyze(df: pd.DataFrame) -> None:
    gate = df["SPY_trend"] & df["QQQ_trend"] & df["SPY_short"] & df["QQQ_short"]
    b_all = df["nh"] > df["nl"]
    b_liq = df["nh_liq"] > df["nl_liq"]
    d = df.dropna(subset=["qqq_fwd63"])
    gate, b_all, b_liq = gate.loc[d.index], b_all.loc[d.index], b_liq.loc[d.index]

    def row(name: str, m: pd.Series) -> None:
        s = d[m.fillna(False).astype(bool)]
        if not len(s):
            print(f"{name:34s} 0일")
            return
        print(f"{name:34s} {len(s):4d}일 | 1M {s.qqq_fwd21.mean():+.2%} (상승 {(s.qqq_fwd21 > 0).mean():.0%}) | "
              f"3M {s.qqq_fwd63.mean():+.2%} (상승 {(s.qqq_fwd63 > 0).mean():.0%}) | "
              f"3M 내 최대낙폭 평균 {s.qqq_dd63.mean():+.1%} · −10%↓ {(s.qqq_dd63 <= -0.10).mean():.0%}")

    print(f"기간 {d.index[0].date()} ~ {d.index[-1].date()} ({len(d)}거래일)")
    row("전체", pd.Series(True, index=d.index))
    row("게이트 차단 (0-1/0-1b 실패)", ~gate)
    row("게이트 통과", gate)
    row("  └ 0-3(전체 종목) 통과", gate & b_all)
    row("  └ 0-3(전체 종목) 실패", gate & ~b_all)
    row("  └ 0-3(유동성 종목) 통과", gate & b_liq)
    row("  └ 0-3(유동성 종목) 실패", gate & ~b_liq)
    row("  └ 전체 실패 & 유동성 통과", gate & ~b_all & b_liq)
    row("  └ 둘 다 실패", gate & ~b_all & ~b_liq)
    # 0-3 실패 구간(연속 일수) 목록 — 게이트 통과일 기준
    fail = (gate & ~b_all).astype(int)
    runs = (fail.diff().fillna(fail.iloc[0]) == 1).cumsum() * fail
    print("\n게이트 통과 중 0-3(전체) 실패 구간 (5거래일 이상):")
    for k, g in d[runs > 0].groupby(runs[runs > 0]):
        if len(g) < 5:
            continue
        print(f"  {g.index[0].date()} ~ {g.index[-1].date()} ({len(g)}일) | 유동성 기준도 실패 "
              f"{(~b_liq.loc[g.index]).mean():.0%} | 시작일 이후 QQQ 3M {g.qqq_fwd63.iloc[0]:+.1%} · "
              f"최대낙폭 {g.qqq_dd63.iloc[0]:+.1%}")


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1])
    if cfg.get("load"):
        data = pd.read_csv(cfg["load"], index_col=0, parse_dates=True)
        for col in ("SPY_trend", "QQQ_trend", "SPY_short", "QQQ_short"):
            data[col] = data[col].astype(bool)
    else:
        data = build(cfg)
        data.to_csv(cfg.get("out", "/tmp/breadth_daily.csv"))
    analyze(data)
