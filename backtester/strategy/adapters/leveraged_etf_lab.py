"""레버리지 ETF(TQQQ·SOXL) 사고팔기 전략 랩 — "장기보유보다 나은가"를 전략별로 같은 자로 잰다.

검증 대상 (docs/strategy_notes/레버리지ETF_전략_검증_2026-10.md):
  ① 장기보유 (기준선)
  ② 추세 필터 — 신호 지수(QQQ 등)가 N일 SMA 위면 보유, 아래면 현금 (Gayed & Bilello 2016 "Leverage for the Long Run").
     진입/이탈 밴드(휩소 완화)와 주봉 판정 옵션.
  ③ 변동성 타기팅 — 비중 = min(상한, 목표 변동성 ÷ 최근 n일 실현 변동성). 레버리지 ETF 손실은 "하락"보다
     "고변동성 구간"에서 나온다(변동성 끌림 ∝ σ²)는 데서 출발.
  ④ ② × ③ 결합.
  ⑤ 9Sig (Jason Kelly) — 주식 60 : 현금 40 으로 시작, 분기마다 "주식 목표치 = 직전 목표 × 1.09" 로 초과분 매도·
     부족분 매수. 라오어 VR과 같은 '밸류패스 추종' 계열(VR 본체는 28 페이지).

공통 시뮬레이터 :func:`simulate_exposure` 는 "매일 종가에 목표 비중을 맞추는" 원장(주식 수·평단·현금)으로,
토스 수수료 + 양도세(연 정산, 기본공제) + 현금 이자를 반영한다. 신호는 종가로 판정하고 같은 종가에 체결
(``exec_lag=0``, 이 저장소의 24·28 페이지와 동일) — 보수적으로 보려면 ``exec_lag=1`` (다음날 종가 체결).

순수 pandas — Streamlit/네트워크 의존 없음. 페이지·CLI가 데이터를 넣어 준다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from strategy.adapters.risk_metrics import compute_risk_metrics

TRADING_DAYS = 252

# 위기 구간 — "장기보유 대비 얼마나 덜 잃었나"를 보는 창. (고점 → 저점)
CRISIS_WINDOWS: dict[str, tuple[date, date]] = {
    "닷컴 붕괴": (date(2000, 3, 10), date(2002, 10, 9)),
    "금융위기": (date(2007, 10, 31), date(2009, 3, 9)),
    "코로나": (date(2020, 2, 19), date(2020, 3, 23)),
    "2022 긴축": (date(2021, 11, 19), date(2022, 12, 28)),
    "2025 관세": (date(2025, 2, 19), date(2025, 4, 8)),
}


@dataclass(frozen=True)
class CostParams:
    """수수료·세금·현금 이자. 기본값 = 토스증권 미국주식 + 양도세 22%(연 250만 공제)."""

    buy_fee: float = 0.001
    sell_fee: float = 0.001 + 0.0000229
    tax_rate: float = 0.22
    tax_deduction: float = 2_500_000.0
    initial_capital: float = 100_000_000.0
    rebalance_band: float = 0.0   # 목표 비중과 현재 비중 차이가 이 이하(자본 대비)면 거래 안 함 — 잔거래·세금 억제.
                                  # 0 이면 현금 이자 재투자 같은 미세 거래도 '매매' 로 세어진다 (CLI 기본 0.02, 페이지 기본 0.05)
    exec_lag: int = 0             # 0 = 신호 종가에 체결, 1 = 다음날 종가


@dataclass
class LabResult:
    name: str
    equity: pd.Series                 # 일별 평가액 (세전, 연말 세금은 차감됨)
    exposure: pd.Series               # 실제 보유 비중 (주식 평가액 / 평가액)
    target: pd.Series                 # 목표 비중 (신호)
    trades: int
    fees: float
    tax: float
    final_after_liquidation: float    # 마지막 날 전량 매도 후 세금까지 낸 값
    metrics: dict = field(default_factory=dict)
    note: str = ""

    @property
    def time_in_market(self) -> float:
        return float((self.exposure > 0.01).mean()) if len(self.exposure) else 0.0


# ----------------------------------------------------------------------------- 신호 (목표 비중)

def sma_filter(signal_close: pd.Series, n: int = 200, enter_band: float = 0.0, exit_band: float = 0.0,
               weekly: bool = False) -> pd.Series:
    """N일 SMA 추세 필터 → 0/1. 히스테리시스: 종가 > SMA×(1+enter) 면 진입, < SMA×(1−exit) 면 이탈, 사이면 유지.

    ``weekly=True`` 면 주의 마지막 거래일에만 판정하고 그 사이는 유지(매매 횟수·세금 절감 변형).
    SMA 가 계산되지 않는 초반 n−1 일은 0(현금).
    """
    s = signal_close.dropna()
    sma = s.rolling(n).mean()
    up = s > sma * (1.0 + enter_band)
    down = s < sma * (1.0 - exit_band)
    state = np.zeros(len(s), dtype=float)
    cur = 0.0
    last_flag = pd.Series(True, index=s.index)
    if weekly:   # 주의 마지막 거래일에만 True
        wk = s.index.to_period("W-FRI")
        last_days = pd.Series(s.index, index=s.index).groupby(wk).last().values
        last_flag = pd.Series(False, index=s.index)
        last_flag.loc[last_days] = True
    for i in range(len(s)):
        if pd.isna(sma.iloc[i]):
            state[i] = 0.0
            continue
        decide = (not weekly) or bool(last_flag.iloc[i])
        if decide:
            if up.iloc[i]:
                cur = 1.0
            elif down.iloc[i]:
                cur = 0.0
        state[i] = cur
    return pd.Series(state, index=s.index, name=f"sma{n}")


def deep_correction_filter(signal_close: pd.Series, n: int = 200, dd: float = 0.15, reenter_n: int = 20,
                           off_fraction: float = 0.0) -> pd.Series:
    """⑩ 급락 회피 — '큰 조정일 때만' 빠진다.

    이탈: 종가 < N일 SMA **그리고** 52주 고점 대비 낙폭 > dd. 재진입: 종가 > reenter_n 일 SMA (빠른 복귀 → V 자 반등을 덜 놓침).
    이탈 중에는 ``off_fraction`` 만 남긴다 (0.33 ≈ 레버리지 ETF 로 1배 지수 노출). 시작 상태는 보유.
    2026-10 격자 탐색(regime_search.py)에서 82개 레짐 규칙 중 세전으로 유일하게 TQQQ 2010~ 장기보유를 넘긴 계열
    (dd 15~18%: 345~489x vs 354x, MDD −61~−68% vs −82%; 1999~ 228~303x vs 1.0x). dd 10~12% 또는 20% 는 휩소로 절반 이하 —
    파라미터에 민감하다는 뜻이므로 15~18% 를 한 묶음으로 본다. 세후(22%)로는 2010~ 장기보유의 55~85%.
    """
    c = signal_close.dropna()
    sma = c.rolling(n).mean()
    drawdown = c / c.rolling(252, min_periods=20).max() - 1.0
    off = ((c < sma) & (drawdown < -dd)).values
    on = (c > c.rolling(reenter_n).mean()).values
    state = np.ones(len(c)); cur = 1.0
    for i in range(len(c)):
        if np.isnan(sma.iloc[i]):
            state[i] = 0.0; continue
        if off[i]:
            cur = 0.0
        elif on[i]:
            cur = 1.0
        state[i] = cur
    out = pd.Series(state, index=c.index, name=f"deepdd{n}_{int(dd * 100)}_re{reenter_n}")
    return out.clip(lower=off_fraction) if off_fraction > 0 else out


def month_end_only(signal: pd.Series) -> pd.Series:
    """일별 신호를 각 달 마지막 거래일 값으로 고정해 다음 달 내내 유지 (Faber 식 월 1회 판정). 선행 참조 없음."""
    last_days = signal.groupby(signal.index.to_period("M")).apply(lambda x: x.index[-1]).values
    return signal.loc[last_days].reindex(signal.index, method="ffill").fillna(0.0)


def realized_vol(close: pd.Series, lookback: int = 20) -> pd.Series:
    """연환산 실현 변동성 (일간 로그수익률 표준편차 × √252)."""
    r = np.log(close / close.shift(1))
    return r.rolling(lookback).std() * math.sqrt(TRADING_DAYS)


def vol_target(lev_close: pd.Series, target_vol: float = 0.45, lookback: int = 20,
               max_exposure: float = 1.0, min_exposure: float = 0.0) -> pd.Series:
    """변동성 타기팅 비중 = clip(목표 변동성 / 실현 변동성, min, max). 변동성 미계산 구간은 0."""
    rv = realized_vol(lev_close, lookback)
    w = (target_vol / rv).clip(lower=min_exposure, upper=max_exposure)
    return w.fillna(0.0).rename("vol_target")


def combine(*signals: pd.Series) -> pd.Series:
    """목표 비중 곱 (필터 0/1 × 타기팅 비중)."""
    out = None
    for s in signals:
        out = s if out is None else out.mul(s.reindex(out.index).ffill().fillna(0.0))
    return out.rename("combined")


# ----------------------------------------------------------------------------- 원장 시뮬레이터

class _Ledger:
    def __init__(self, c: CostParams):
        self.c = c
        self.cash = c.initial_capital
        self.shares = 0.0
        self.avg_cost = 0.0
        self.realized_ytd = 0.0
        self.fees = 0.0
        self.tax = 0.0
        self.trades = 0

    def equity(self, px: float) -> float:
        return self.cash + self.shares * px

    def buy(self, notional: float, px: float) -> None:
        if notional <= 0:
            return
        cost = notional * (1.0 + self.c.buy_fee)
        if cost > self.cash:
            notional = self.cash / (1.0 + self.c.buy_fee)
            cost = self.cash
        if notional <= 0:
            return
        new_shares = notional / px
        self.avg_cost = ((self.shares * self.avg_cost) + notional) / (self.shares + new_shares)
        self.shares += new_shares
        self.cash -= cost
        self.fees += cost - notional
        self.trades += 1

    def sell(self, notional: float, px: float) -> None:
        if notional <= 0 or self.shares <= 0:
            return
        notional = min(notional, self.shares * px)
        sold = notional / px
        self.realized_ytd += notional - sold * self.avg_cost
        self.shares -= sold
        proceeds = notional * (1.0 - self.c.sell_fee)
        self.cash += proceeds
        self.fees += notional - proceeds
        self.trades += 1
        if self.shares < 1e-12:
            self.shares, self.avg_cost = 0.0, 0.0

    def settle_tax(self) -> None:
        due = max(0.0, self.realized_ytd - self.c.tax_deduction) * self.c.tax_rate
        self.cash -= due
        self.tax += due
        self.realized_ytd = 0.0

    def liquidate_value(self, px: float) -> float:
        """전량 매도 + 세금 낸 뒤 현금 (원장은 건드리지 않음)."""
        notional = self.shares * px
        realized = self.realized_ytd + (notional - self.shares * self.avg_cost)
        due = max(0.0, realized - self.c.tax_deduction) * self.c.tax_rate
        return self.cash + notional * (1.0 - self.c.sell_fee) - due


def _cash_rate_daily(cash_rate: pd.Series | float | None, index: pd.DatetimeIndex) -> pd.Series:
    if cash_rate is None:
        return pd.Series(0.0, index=index)
    if isinstance(cash_rate, (int, float)):
        return pd.Series(float(cash_rate) / TRADING_DAYS, index=index)
    return (cash_rate.reindex(index).ffill().fillna(0.0) / TRADING_DAYS)


def simulate_exposure(price: pd.Series, target: pd.Series, costs: CostParams = CostParams(),
                      cash_rate: pd.Series | float | None = None, name: str = "") -> LabResult:
    """목표 비중 시계열대로 매일 종가에 맞춰 사고판다.

    ``target`` 은 ``price`` 인덱스 기준으로 재정렬(ffill)되고 ``exec_lag`` 만큼 뒤로 민다.
    ``cash_rate`` 는 연환산(소수) — 상수 또는 시계열(^IRX/100 등). 현금에만 붙는다.
    """
    px = price.dropna()
    tgt = target.reindex(px.index).ffill().fillna(0.0).clip(0.0, 1.0)
    if costs.exec_lag > 0:
        tgt = tgt.shift(costs.exec_lag).fillna(0.0)
    rate = _cash_rate_daily(cash_rate, px.index)
    led = _Ledger(costs)
    eq = np.empty(len(px)); ex = np.empty(len(px))
    years = px.index.year
    for i, (ts, p) in enumerate(px.items()):
        p = float(p)
        if i > 0:
            led.cash *= 1.0 + float(rate.iloc[i])
        equity = led.equity(p)
        want = float(tgt.iloc[i]) * equity
        have = led.shares * p
        diff = want - have
        if abs(diff) > costs.rebalance_band * equity and abs(diff) > 1e-9:
            if diff > 0:
                led.buy(diff, p)
            else:
                led.sell(-diff, p)
        equity = led.equity(p)
        if i == len(px) - 1 or years[i + 1] != years[i]:
            led.settle_tax()
            equity = led.equity(p)
        eq[i] = equity
        ex[i] = (led.shares * p) / equity if equity > 0 else 0.0
    equity_s = pd.Series(eq, index=px.index, name=name or "equity")
    res = LabResult(name=name, equity=equity_s, exposure=pd.Series(ex, index=px.index), target=tgt,
                    trades=led.trades, fees=led.fees, tax=led.tax,
                    final_after_liquidation=led.liquidate_value(float(px.iloc[-1])))
    res.metrics = compute_risk_metrics(equity_s)
    return res


def simulate_nine_sig(price: pd.Series, costs: CostParams = CostParams(), cash_rate: pd.Series | float | None = None,
                      stock_weight: float = 0.6, quarterly_growth: float = 0.09, freq: str = "Q",
                      name: str = "9Sig") -> LabResult:
    """9Sig: 주식 목표치 V 를 분기마다 ×(1+g). 평가액 > V 면 초과분 매도(현금), < V 면 부족분을 현금 한도로 매수.

    원전은 채권 ETF(AGG) 슬리브지만 여기선 현금(+이자)으로 둔다 — 2022년처럼 채권이 같이 빠지는 해를
    낙관하지 않기 위한 보수적 단순화. ``freq``: "Q"(분기) 또는 "M"(월, g 는 분기 기준값을 1/3 로 환산).
    """
    px = price.dropna()
    rate = _cash_rate_daily(cash_rate, px.index)
    led = _Ledger(costs)
    g = quarterly_growth if freq == "Q" else (1.0 + quarterly_growth) ** (1 / 3) - 1.0
    periods = px.index.to_period(freq)
    eq = np.empty(len(px)); ex = np.empty(len(px)); tg = np.empty(len(px))
    years = px.index.year
    V = None
    for i, (ts, p) in enumerate(px.items()):
        p = float(p)
        if i > 0:
            led.cash *= 1.0 + float(rate.iloc[i])
        if V is None:
            led.buy(stock_weight * led.equity(p), p)
            V = led.shares * p
        elif periods[i] != periods[i - 1]:
            V *= 1.0 + g
            have = led.shares * p
            if have > V:
                led.sell(have - V, p)
            elif have < V:
                led.buy(V - have, p)
        equity = led.equity(p)
        if i == len(px) - 1 or years[i + 1] != years[i]:
            led.settle_tax()
            equity = led.equity(p)
        eq[i] = equity
        ex[i] = (led.shares * p) / equity if equity > 0 else 0.0
        tg[i] = V / equity if equity > 0 else 0.0
    equity_s = pd.Series(eq, index=px.index, name=name)
    res = LabResult(name=name, equity=equity_s, exposure=pd.Series(ex, index=px.index),
                    target=pd.Series(tg, index=px.index), trades=led.trades, fees=led.fees, tax=led.tax,
                    final_after_liquidation=led.liquidate_value(float(px.iloc[-1])))
    res.metrics = compute_risk_metrics(equity_s)
    return res


# ----------------------------------------------------------------------------- 묶음 실행 / 비교

@dataclass(frozen=True)
class LabParams:
    sma_n: int = 200
    sma_enter: float = 0.0
    sma_exit: float = 0.0
    sma_weekly: bool = False
    vol_target: float = 0.45
    vol_lookback: int = 20
    vol_max: float = 1.0
    sig_stock_weight: float = 0.6
    sig_growth: float = 0.09
    sig_freq: str = "Q"
    dd_n: int = 200
    dd_depth: float = 0.15
    dd_reenter: int = 20
    dd_off_fraction: float = 0.0
    sma_monthly: bool = False       # ② 를 월말에만 판정


STRATEGY_ORDER = ["① 장기보유", "② 추세 필터", "③ 변동성 타기팅", "④ 필터+타기팅", "⑤ 9Sig", "⑩ 급락 회피"]


def run_lab(lev_close: pd.Series, signal_close: pd.Series, p: LabParams = LabParams(),
            costs: CostParams = CostParams(), cash_rate: pd.Series | float | None = None,
            start: date | None = None) -> dict[str, LabResult]:
    """5개 전략을 같은 가격·비용으로 돌린다. ``signal_close`` 는 SMA 워밍업용으로 ``lev_close`` 보다 앞서 시작해도 된다.

    ``start`` 를 주면 그날부터 평가(그 이전은 SMA·변동성 워밍업에만 사용).
    """
    lev = lev_close.dropna()
    sig = signal_close.dropna()
    f = sma_filter(sig, p.sma_n, p.sma_enter, p.sma_exit, p.sma_weekly)
    if p.sma_monthly:
        f = month_end_only(f)
    f = f.reindex(lev.index).ffill().fillna(0.0)
    v = vol_target(lev, p.vol_target, p.vol_lookback, p.vol_max)
    d = deep_correction_filter(sig, p.dd_n, p.dd_depth, p.dd_reenter, p.dd_off_fraction).reindex(lev.index).ffill().fillna(0.0)
    if start is not None:
        lev = lev[lev.index >= pd.Timestamp(start)]
        f, v, d = f.reindex(lev.index), v.reindex(lev.index), d.reindex(lev.index)
    one = pd.Series(1.0, index=lev.index)
    out = {
        STRATEGY_ORDER[0]: simulate_exposure(lev, one, costs, cash_rate, STRATEGY_ORDER[0]),
        STRATEGY_ORDER[1]: simulate_exposure(lev, f, costs, cash_rate, STRATEGY_ORDER[1]),
        STRATEGY_ORDER[2]: simulate_exposure(lev, v, costs, cash_rate, STRATEGY_ORDER[2]),
        STRATEGY_ORDER[3]: simulate_exposure(lev, combine(f, v), costs, cash_rate, STRATEGY_ORDER[3]),
        STRATEGY_ORDER[4]: simulate_nine_sig(lev, costs, cash_rate, p.sig_stock_weight, p.sig_growth, p.sig_freq,
                                             STRATEGY_ORDER[4]),
        STRATEGY_ORDER[5]: simulate_exposure(lev, d, costs, cash_rate, STRATEGY_ORDER[5]),
    }
    return out


def summary_table(results: dict[str, LabResult]) -> pd.DataFrame:
    rows = []
    for k, r in results.items():
        m = r.metrics
        rows.append({
            "전략": k, "최종 배수": r.equity.iloc[-1] / r.equity.iloc[0], "CAGR": m["cagr"], "MDD": m["mdd"],
            "최악 연도": m["worst_year"], "연 변동성": m["ann_vol"], "샤프": m["sharpe"], "칼마": m["calmar"],
            "최장 수면(일)": m["longest_underwater_days"], "매매 횟수": r.trades, "시장 노출": r.time_in_market,
            "수수료+세금": (r.fees + r.tax) / r.equity.iloc[0], "청산 후 배수": r.final_after_liquidation / r.equity.iloc[0],
        })
    return pd.DataFrame(rows)


def yearly_returns(results: dict[str, LabResult]) -> pd.DataFrame:
    cols = {}
    for k, r in results.items():
        y = r.equity.groupby(r.equity.index.year).agg(["first", "last"])
        cols[k] = y["last"] / y["first"] - 1.0
    return pd.DataFrame(cols)


def crisis_table(results: dict[str, LabResult], windows: dict[str, tuple[date, date]] = CRISIS_WINDOWS) -> pd.DataFrame:
    """위기 창(고점→저점)별 전략 수익률. 데이터가 창을 덮지 못하면 NaN."""
    rows = []
    for label, (a, b) in windows.items():
        row = {"구간": label, "시작": a, "끝": b}
        for k, r in results.items():
            e = r.equity
            if e.index[0] > pd.Timestamp(a) or e.index[-1] < pd.Timestamp(b):
                row[k] = np.nan
                continue
            ea = e[e.index <= pd.Timestamp(a)].iloc[-1]
            eb = e[e.index <= pd.Timestamp(b)].iloc[-1]
            row[k] = eb / ea - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def trade_log(res: LabResult, min_change: float = 0.05) -> pd.DataFrame:
    """비중이 ``min_change`` 이상 바뀐 날만 추려 매매 일지로. (일별 리밸런싱 전략은 길어질 수 있음)"""
    ex = res.exposure
    d = ex.diff().fillna(ex)
    pts = d[d.abs() >= min_change]
    return pd.DataFrame({"날짜": pts.index.date, "비중 변화": pts.values, "비중": ex.loc[pts.index].values,
                         "평가액": res.equity.loc[pts.index].values})


def current_signal(lev_close: pd.Series, signal_close: pd.Series, p: LabParams = LabParams()) -> dict:
    """마지막 봉 기준 '지금 각 전략이 뭐라 하나'."""
    sig = signal_close.dropna(); lev = lev_close.dropna()
    sma = sig.rolling(p.sma_n).mean()
    f = sma_filter(sig, p.sma_n, p.sma_enter, p.sma_exit, p.sma_weekly)
    rv = realized_vol(lev, p.vol_lookback)
    w = float(min(p.vol_max, p.vol_target / rv.iloc[-1])) if rv.iloc[-1] > 0 else 0.0
    return {
        "as_of": lev.index[-1].date(),
        "signal_close": float(sig.iloc[-1]), "sma": float(sma.iloc[-1]),
        "dist_to_sma": float(sig.iloc[-1] / sma.iloc[-1] - 1.0),
        "filter_on": bool(f.iloc[-1] > 0.5),
        "realized_vol": float(rv.iloc[-1]), "vol_weight": w,
        "combined_weight": w if f.iloc[-1] > 0.5 else 0.0,
    }


def param_grid(lev_close: pd.Series, signal_close: pd.Series, grid: list[LabParams], costs: CostParams = CostParams(),
               cash_rate: pd.Series | float | None = None, start: date | None = None,
               which: str = STRATEGY_ORDER[1]) -> pd.DataFrame:
    """파라미터 민감도 — 한 전략을 여러 파라미터로 돌려 CAGR/MDD/매매 횟수를 한 표로. 과최적화 점검용."""
    rows = []
    for p in grid:
        r = run_lab(lev_close, signal_close, p, costs, cash_rate, start)[which]
        rows.append({"sma_n": p.sma_n, "enter": p.sma_enter, "exit": p.sma_exit, "weekly": p.sma_weekly,
                     "vol_target": p.vol_target, "lookback": p.vol_lookback,
                     "CAGR": r.metrics["cagr"], "MDD": r.metrics["mdd"], "샤프": r.metrics["sharpe"],
                     "매매": r.trades, "최종 배수": r.equity.iloc[-1] / r.equity.iloc[0]})
    return pd.DataFrame(rows)
