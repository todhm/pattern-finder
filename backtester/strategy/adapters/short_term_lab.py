"""단기(2~10일 보유) 고수 전략 4개 — 코너스 RSI(2)·Double 7s, 라시케 Holy Grail·Turtle Soup.

레버리지 ETF 랩(leveraged_etf_lab)의 원장 시뮬레이터에 "목표 비중 0/1" 신호로 넣는다. 신호는 종가로 판정하고
같은 종가에 체결(lag 0) — 책의 "다음날 시가/돌파 체결" 대신 종가 근사다 (보수적으로 보려면 CostParams.exec_lag=1).

출처
- Connors & Alvarez, *Short Term Trading Strategies That Work* (2008): RSI(2) — 200일선 위에서 2일 RSI < 10 매수, 종가 > 5일 SMA 청산.
  Double 7s — 200일선 위에서 종가가 7일 최저 종가면 매수, 7일 최고 종가면 청산.
- Raschke & Connors, *Street Smarts* (1995): Holy Grail — ADX(14) > 30 상승 추세, 20일 EMA 되돌림 터치 후 직전 봉 고가 돌파 매수,
  직전 스윙 고점 도달 청산, 되돌림 저점 이탈 손절. Turtle Soup — 20일 신저가 가짜 돌파(직전 20일 저점이 4일 이상 전) 후
  그 저점 위로 복귀하면 매수, 당일 저가 이탈 손절, 2~6일 안에 익절.
모두 **롱만** (TQQQ·SOXL 보유자 관점). 숏 버전은 구현하지 않았다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import CostParams, LabResult, simulate_exposure


@dataclass(frozen=True)
class ShortTermParams:
    sma_trend: int = 200
    rsi_len: int = 2
    rsi_buy: float = 10.0
    rsi_exit_sma: int = 5
    d7_len: int = 7
    adx_len: int = 14
    adx_min: float = 30.0
    hg_ema: int = 20
    hg_max_hold: int = 10
    hg_swing: int = 20
    ts_len: int = 20
    ts_min_gap: int = 4
    ts_max_hold: int = 5
    ts_plus_one: bool = False      # True = 다음날 종가가 저점 위면 진입 (Turtle Soup Plus One)


SHORT_TERM_ORDER = ["⑥ 코너스 RSI(2)", "⑦ 코너스 Double 7s", "⑧ 라시케 Holy Grail", "⑨ 라시케 Turtle Soup"]


# ----------------------------------------------------------------------------- 지표

def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0.0)


# ----------------------------------------------------------------------------- 신호 (0/1 비중)

def connors_rsi2(close: pd.Series, p: ShortTermParams = ShortTermParams()) -> pd.Series:
    sma200 = close.rolling(p.sma_trend).mean()
    r = rsi(close, p.rsi_len)
    sma5 = close.rolling(p.rsi_exit_sma).mean()
    state = np.zeros(len(close)); cur = 0.0
    for i in range(len(close)):
        if np.isnan(sma200.iloc[i]):
            continue
        if cur == 0.0 and close.iloc[i] > sma200.iloc[i] and r.iloc[i] < p.rsi_buy:
            cur = 1.0
        elif cur == 1.0 and close.iloc[i] > sma5.iloc[i]:
            cur = 0.0
        state[i] = cur
    return pd.Series(state, index=close.index, name="rsi2")


def connors_double7(close: pd.Series, p: ShortTermParams = ShortTermParams()) -> pd.Series:
    sma200 = close.rolling(p.sma_trend).mean()
    lo7 = close.rolling(p.d7_len).min()
    hi7 = close.rolling(p.d7_len).max()
    state = np.zeros(len(close)); cur = 0.0
    for i in range(len(close)):
        if np.isnan(sma200.iloc[i]) or np.isnan(lo7.iloc[i]):
            continue
        if cur == 0.0 and close.iloc[i] > sma200.iloc[i] and close.iloc[i] <= lo7.iloc[i]:
            cur = 1.0
        elif cur == 1.0 and close.iloc[i] >= hi7.iloc[i]:
            cur = 0.0
        state[i] = cur
    return pd.Series(state, index=close.index, name="double7")


def raschke_holy_grail(df: pd.DataFrame, p: ShortTermParams = ShortTermParams()) -> pd.Series:
    """ADX(14) > 30 이고 5일 전에 주가가 20EMA 위였던(상승 추세 확립) 상태에서 저가가 20EMA 를 터치한 뒤
    3일 안에 종가가 터치 봉 고가를 넘으면 매수. 청산: 종가 ≥ 되돌림 이전 20일 최고가(목표) / 종가 < 되돌림 저점(손절) / 최대 보유일.
    책의 "ADX 상승 중" 은 되돌림 동안 ADX 가 자연히 꺾이므로 그대로 쓰면 신호가 거의 안 나온다 — 'ADX > 30' 으로 대체."""
    c, h, l = df["Close"], df["High"], df["Low"]
    ema = c.ewm(span=p.hg_ema, adjust=False).mean()
    a = adx(df, p.adx_len)
    swing_hi = h.rolling(p.hg_swing).max().shift(1)
    state = np.zeros(len(c)); cur = 0.0
    touch_i = None; entry_i = None; stop = target = None
    for i in range(p.hg_ema + 6, len(c)):
        if cur == 0.0:
            trend = a.iloc[i] > p.adx_min and c.iloc[i - 5] > ema.iloc[i - 5] and c.iloc[i] > ema.iloc[i] * 0.97
            if trend and l.iloc[i] <= ema.iloc[i]:
                touch_i = i
            if touch_i is not None and i - touch_i <= 3 and i > touch_i and c.iloc[i] > h.iloc[touch_i]:
                cur = 1.0; entry_i = i
                stop = float(l.iloc[touch_i:i + 1].min()); target = float(swing_hi.iloc[touch_i])
                touch_i = None
        else:
            if c.iloc[i] >= target or c.iloc[i] < stop or i - entry_i >= p.hg_max_hold:
                cur = 0.0
        state[i] = cur
    return pd.Series(state, index=c.index, name="holy_grail")


def raschke_turtle_soup(df: pd.DataFrame, p: ShortTermParams = ShortTermParams()) -> pd.Series:
    """오늘 저가 < 직전 20일 최저가(그 최저가 날짜가 ≥ 4일 전) 이고 종가가 그 최저가 위로 복귀 → 매수.
    청산: 종가 < 진입일 저가(손절) / 최대 보유일(익절). plus_one 이면 다음날 종가 기준으로 진입 판정."""
    c, l = df["Close"], df["Low"]
    n = p.ts_len
    state = np.zeros(len(c)); cur = 0.0
    entry_i = None; stop = None; pending = None
    for i in range(n + 1, len(c)):
        if cur == 0.0:
            window = l.iloc[i - n:i]
            prior_low = float(window.min()); prior_pos = int(window.values.argmin())
            days_since = n - prior_pos
            if pending is not None:
                if c.iloc[i] > pending:
                    cur = 1.0; entry_i = i; stop = float(l.iloc[i - 1])
                pending = None
            elif l.iloc[i] < prior_low and days_since >= p.ts_min_gap:
                if p.ts_plus_one:
                    pending = prior_low
                elif c.iloc[i] > prior_low:
                    cur = 1.0; entry_i = i; stop = float(l.iloc[i])
        else:
            if c.iloc[i] < stop or i - entry_i >= p.ts_max_hold:
                cur = 0.0
        state[i] = cur
    return pd.Series(state, index=c.index, name="turtle_soup")


# ----------------------------------------------------------------------------- 실행 / 통계

def run_short_term(lev_close: pd.Series, signal_df: pd.DataFrame, p: ShortTermParams = ShortTermParams(),
                   costs: CostParams = CostParams(), cash_rate: pd.Series | float | None = None,
                   start: date | None = None) -> dict[str, LabResult]:
    """4개 전략을 같은 가격·비용으로. ``signal_df`` = 신호 지수 OHLC (SMA200 워밍업 포함), ``lev_close`` = 거래 ETF 종가."""
    sig = signal_df.dropna(subset=["Close"])
    sigs = {
        SHORT_TERM_ORDER[0]: connors_rsi2(sig["Close"], p),
        SHORT_TERM_ORDER[1]: connors_double7(sig["Close"], p),
        SHORT_TERM_ORDER[2]: raschke_holy_grail(sig, p),
        SHORT_TERM_ORDER[3]: raschke_turtle_soup(sig, p),
    }
    lev = lev_close.dropna()
    if start is not None:
        lev = lev[lev.index >= pd.Timestamp(start)]
    out = {}
    for k, s in sigs.items():
        tgt = s.reindex(lev.index).ffill().fillna(0.0)
        out[k] = simulate_exposure(lev, tgt, costs, cash_rate, k)
    return out


def trade_stats(res: LabResult, price: pd.Series) -> dict:
    """비중 0→1→0 구간을 거래 하나로 보고 거래 수·승률·평균 보유일·평균 수익률(가격 기준, 비용 전)."""
    ex = (res.exposure > 0.5).astype(int)
    d = ex.diff().fillna(ex)
    entries = list(d[d == 1].index); exits = list(d[d == -1].index)
    px = price.reindex(res.exposure.index).ffill()
    rows = []
    for e in entries:
        x = next((t for t in exits if t > e), res.exposure.index[-1])
        rows.append({"entry": e, "exit": x, "days": int(((res.exposure.index > e) & (res.exposure.index <= x)).sum()),
                     "ret": float(px.loc[x] / px.loc[e] - 1.0)})
    if not rows:
        return {"trades": 0, "win_rate": np.nan, "avg_days": np.nan, "avg_ret": np.nan, "per_year": 0.0}
    t = pd.DataFrame(rows)
    years = max((res.exposure.index[-1] - res.exposure.index[0]).days / 365.25, 1e-9)
    return {"trades": int(len(t)), "win_rate": float((t["ret"] > 0).mean()), "avg_days": float(t["days"].mean()),
            "avg_ret": float(t["ret"].mean()), "median_ret": float(t["ret"].median()), "per_year": len(t) / years,
            "worst": float(t["ret"].min()), "best": float(t["ret"].max())}
