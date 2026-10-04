"""Gayed «Leverage for the Long Run»(2016, Dow Award 논문) 식 레버리지 로테이션을 TQQQ 에 적용.

규칙: 기초지수(QQQ) 일봉 종가 > N일 SMA 이면 다음 날부터 3x(TQQQ) 100%, 아니면 현금.
와인스타인 30주 MA 의 일봉 버전이자, "레버리지 ETF 는 B&H 가 아니라 추세 위에서만 들어라" 는 가장 유명한 공개 규칙.

실행: docker compose exec backtester python gayed_sma200_tqqq.py
출력: 실제 TQQQ(2011~) 와 합성 3x(2000~) 두 표본에서 SMA 50/100/200, 확인일수 1/3 변형의 CAGR·MDD vs 보유.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd

from data.adapters.cached_market_data import CachedMarketDataAdapter
from data.adapters.yfinance_adapter import YFinanceAdapter

warnings.filterwarnings("ignore")
md = CachedMarketDataAdapter(YFinanceAdapter())
SLIP = 0.0005  # 일간 전환이라 편도 5bp


def fetch(sym, start=date(1998, 1, 1)):
    df = md.fetch_ohlcv(sym, start, date.today())
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def synth3x(qqq):
    r = qqq["Close"].pct_change().fillna(0) * 3 - 0.0095 / 252
    return 100 * (1 + r).cumprod()


def stats(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = eq / eq.cummax() - 1
    return cagr, dd.min(), dd.idxmin().date()


def rotate(qqq_close, lev_close, n, confirm=1, start=None):
    sma = qqq_close.rolling(n).mean()
    above = (qqq_close > sma).astype(int)
    sig = above.rolling(confirm).min() if confirm > 1 else above          # confirm 일 연속 위여야 진입
    below = (qqq_close < sma).astype(int)
    out = below.rolling(confirm).min() if confirm > 1 else below          # confirm 일 연속 아래여야 청산
    pos = pd.Series(np.nan, index=qqq_close.index)
    pos[sig == 1] = 1.0
    pos[out == 1] = 0.0
    pos = pos.ffill().fillna(0.0).shift(1).fillna(0.0)                   # 다음 날부터 적용
    r = lev_close.pct_change().fillna(0)
    turn = pos.diff().abs().fillna(0)
    strat = (1 + pos * r - turn * SLIP).cumprod()
    if start:
        strat = strat.loc[start:]
        strat = strat / strat.iloc[0]
    return strat, pos


def main():
    qqq, tqqq = fetch("QQQ"), fetch("TQQQ")
    syn = synth3x(qqq)
    print("# Gayed 식 SMA 로테이션 × TQQQ")
    for label, lev, start in (("실제 TQQQ 2011-01~", tqqq["Close"], "2011-01-03"), ("합성 3x 2000-01~ (닷컴·2008 포함)", syn, "2000-01-03")):
        q = qqq["Close"].reindex(lev.index).ffill() if label.startswith("실제") else qqq["Close"]
        lev = lev.loc[q.index]
        bh = lev.loc[start:] / lev.loc[start:].iloc[0]
        qbh = q.loc[start:] / q.loc[start:].iloc[0]
        print(f"\n## {label}")
        c, m, md_ = stats(bh)
        print(f"- 3x 보유: CAGR {c:+.1%}, MDD {m:+.1%} ({md_})")
        c, m, md_ = stats(qbh)
        print(f"- QQQ 보유: CAGR {c:+.1%}, MDD {m:+.1%} ({md_})")
        rows = []
        for n in (50, 100, 150, 200):
            for conf in (1, 3, 5):
                eq, pos = rotate(q, lev, n, conf, start)
                c, m, md_ = stats(eq)
                switches = int(pos.loc[start:].diff().abs().sum())
                rows.append({"SMA": n, "확인일": conf, "CAGR": f"{c:+.1%}", "MDD": f"{m:+.1%}", "MDD 저점": md_, "전환 횟수": switches,
                             "시장 참여": f"{pos.loc[start:].mean():.0%}"})
        print(pd.DataFrame(rows).to_string(index=False))
        # 연도별 (200/3 기준)
        eq, pos = rotate(q, lev, 200, 3, start)
        yr = pd.DataFrame({"전략": eq.resample("YE").last().pct_change(), "3x보유": bh.resample("YE").last().pct_change()}).dropna()
        yr.index = yr.index.year
        print("연도별 (SMA200, 3일 확인):")
        print(yr.map(lambda v: f"{v:+.0%}").T.to_string())


if __name__ == "__main__":
    main()
