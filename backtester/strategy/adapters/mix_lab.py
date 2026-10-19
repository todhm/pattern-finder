"""다자산 혼합 원장 — TQQQ(⑩ 스위치) + 동승자(장기국채 1x/3x, 금) 를 세후·수수료·리밸런싱 규칙으로.

단일 자산 `leveraged_etf_lab._Ledger` 를 자산별 평균단가 원장으로 일반화. 세금은 전 자산 실현손익을 연 합산(250만 공제, 이월 없음).
리밸런싱: 달력(분기/연) 리밸런싱 날 또는 스위치 상태가 바뀐 날에만 목표로 이동, 그 외엔 드리프트.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import TRADING_DAYS, CostParams


@dataclass
class MixResult:
    name: str
    equity: pd.Series
    weights: pd.DataFrame
    trades: int
    fees: float
    tax: float
    final_after_liquidation: float
    metrics: dict = field(default_factory=dict)


class _MultiLedger:
    def __init__(self, c: CostParams, assets: list[str]):
        self.c = c; self.cash = c.initial_capital
        self.shares = {a: 0.0 for a in assets}; self.avg = {a: 0.0 for a in assets}
        self.realized_ytd = 0.0; self.fees = 0.0; self.tax = 0.0; self.trades = 0

    def equity(self, px: dict[str, float]) -> float:
        return self.cash + sum(self.shares[a] * px[a] for a in self.shares)

    def buy(self, a: str, notional: float, p: float) -> None:
        if notional <= 0: return
        cost = notional * (1 + self.c.buy_fee)
        if cost > self.cash:
            notional = self.cash / (1 + self.c.buy_fee); cost = self.cash
        if notional <= 0: return
        n = notional / p
        self.avg[a] = (self.shares[a] * self.avg[a] + notional) / (self.shares[a] + n)
        self.shares[a] += n; self.cash -= cost; self.fees += cost - notional; self.trades += 1

    def sell(self, a: str, notional: float, p: float) -> None:
        if notional <= 0 or self.shares[a] <= 0: return
        notional = min(notional, self.shares[a] * p); n = notional / p
        self.realized_ytd += notional - n * self.avg[a]
        self.shares[a] -= n; proceeds = notional * (1 - self.c.sell_fee)
        self.cash += proceeds; self.fees += notional - proceeds; self.trades += 1
        if self.shares[a] < 1e-12: self.shares[a], self.avg[a] = 0.0, 0.0

    def settle_tax(self) -> None:
        due = max(0.0, self.realized_ytd - self.c.tax_deduction) * self.c.tax_rate
        self.cash -= due; self.tax += due; self.realized_ytd = 0.0

    def liquidate(self, px: dict[str, float]) -> float:
        notional = sum(self.shares[a] * px[a] for a in self.shares)
        realized = self.realized_ytd + sum(self.shares[a] * (px[a] - self.avg[a]) for a in self.shares)
        due = max(0.0, realized - self.c.tax_deduction) * self.c.tax_rate
        return self.cash + notional * (1 - self.c.sell_fee) - due


def simulate_mix(prices: pd.DataFrame, targets: pd.DataFrame, trade_days: pd.Series, costs: CostParams,
                 cash_rate: pd.Series | None, name: str = "") -> MixResult:
    """prices: 자산별 종가(동일 인덱스). targets: 자산별 목표 비중(합 ≤ 1, 나머지 현금). trade_days: 그날 목표로 이동할지(bool).
    exec_lag 는 targets/trade_days 를 하루 미룬다."""
    px = prices.dropna(); assets = list(px.columns)
    tg = targets.reindex(px.index).ffill().fillna(0.0).clip(0, 1)
    td = trade_days.reindex(px.index).fillna(False).astype(bool)
    if costs.exec_lag > 0:
        tg = tg.shift(costs.exec_lag).fillna(0.0); td = td.shift(costs.exec_lag).fillna(False).astype(bool)
    rate = (cash_rate.reindex(px.index).ffill().fillna(0.0) / TRADING_DAYS) if cash_rate is not None else pd.Series(0.0, index=px.index)
    led = _MultiLedger(costs, assets)
    eq = np.empty(len(px)); W = np.zeros((len(px), len(assets))); years = px.index.year
    pv = px.values; tv = tg.values; tdv = td.values.copy(); rv = rate.values
    tdv[min(costs.exec_lag, len(tdv) - 1)] = True  # 표본 첫날(지연 반영)엔 반드시 목표로 이동 — 시작 시점이 리밸런싱 날이 아니어도 현금에 머물지 않게
    for i in range(len(px)):
        p = {a: float(pv[i, j]) for j, a in enumerate(assets)}
        if i > 0: led.cash *= 1 + float(rv[i])
        if tdv[i]:
            E = led.equity(p)
            # 먼저 팔고 나중에 산다 (현금 확보)
            diffs = {a: float(tv[i, j]) * E - led.shares[a] * p[a] for j, a in enumerate(assets)}
            for a, d in diffs.items():
                if d < 0 and abs(d) > costs.rebalance_band * E: led.sell(a, -d, p[a])
            for a, d in diffs.items():
                if d > 0 and abs(d) > costs.rebalance_band * E: led.buy(a, d, p[a])
        E = led.equity(p)
        if i == len(px) - 1 or years[i + 1] != years[i]:
            led.settle_tax(); E = led.equity(p)
        eq[i] = E
        for j, a in enumerate(assets): W[i, j] = led.shares[a] * p[a] / E if E > 0 else 0.0
    equity = pd.Series(eq, index=px.index, name=name)
    peak = equity.cummax(); mdd = float((equity / peak - 1).min())
    return MixResult(name, equity, pd.DataFrame(W, index=px.index, columns=assets), led.trades, led.fees, led.tax,
                     led.liquidate({a: float(pv[-1, j]) for j, a in enumerate(assets)}), {"mdd": mdd})


def calendar_days(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """분기/연 마지막 거래일 True."""
    if freq == "none":
        return pd.Series(False, index=index)
    per = index.to_period({"quarterly": "Q", "annual": "Y", "monthly": "M"}[freq])
    last = pd.Series(index, index=index).groupby(per).last()
    s = pd.Series(False, index=index); s.loc[last.values] = True
    return s


def synthetic_long_bond(yield_pct: pd.Series, duration: float = 12.0, convexity: float = 180.0) -> pd.Series:
    """상수 만기 국채 수익률(%)로 장기국채 1x 총수익 지수 근사: r ≈ y/252 − D·Δy + ½·C·Δy²."""
    y = yield_pct.dropna() / 100.0
    dy = y.diff().fillna(0.0)
    r = y.shift(1).fillna(y.iloc[0]) / TRADING_DAYS - duration * dy + 0.5 * convexity * dy ** 2
    return (1 + r).cumprod() * 100.0
