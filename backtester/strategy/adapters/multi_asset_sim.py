"""다자산 목표 비중 원장 시뮬레이터 — `leveraged_etf_lab.simulate_exposure` 와 같은 규약(수수료·연말 세금 통산·공제·밴드·lag·현금 이자) 을 N 자산으로.

목표 비중은 DataFrame(열 = 자산, 값 0~1, 행 합 ≤ 1), 나머지는 현금. 매일 종가에 맞춰 매도 먼저 → 매수.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.leveraged_etf_lab import TRADING_DAYS, CostParams, LabResult, _cash_rate_daily
from strategy.adapters.risk_metrics import compute_risk_metrics


class _Lot:
    __slots__ = ("shares", "avg")

    def __init__(self):
        self.shares = 0.0; self.avg = 0.0


def simulate_weights(prices: pd.DataFrame, weights: pd.DataFrame, costs: CostParams = CostParams(),
                     cash_rate: pd.Series | float | None = None, name: str = "",
                     dividend_yields: dict[str, float] | None = None) -> LabResult:
    """``dividend_yields`` = 열별 연 배당수익률(없으면 ``costs.dividend_yield``) — 배당세 드래그(H13)."""
    px = prices.dropna(how="all").ffill()
    cols = list(px.columns)
    dy = np.array([(dividend_yields or {}).get(c, costs.dividend_yield) for c in cols]) * costs.dividend_tax_rate / TRADING_DAYS
    w = weights.reindex(px.index).ffill().reindex(columns=cols).fillna(0.0).clip(0.0, 1.0)
    if costs.exec_lag > 0:
        w = w.shift(costs.exec_lag).fillna(0.0)
    rate = _cash_rate_daily(cash_rate, px.index).to_numpy()
    P = px.to_numpy(dtype=float); W = w.to_numpy(dtype=float)
    years = px.index.year.to_numpy()
    cash = costs.initial_capital; lots = [_Lot() for _ in cols]
    realized = 0.0; fees = 0.0; tax = 0.0; trades = 0
    eq = np.empty(len(px)); ex = np.empty(len(px))
    for i in range(len(px)):
        p = P[i]
        if i > 0:
            cash *= 1.0 + rate[i]
            for k, l in enumerate(lots):
                if dy[k] > 0 and l.shares > 0 and not np.isnan(p[k]):
                    dtax = l.shares * p[k] * dy[k]; cash -= dtax; tax += dtax
        equity = cash + sum(l.shares * (p[k] if not np.isnan(p[k]) else 0.0) for k, l in enumerate(lots))
        orders = []
        for k, l in enumerate(lots):
            if np.isnan(p[k]):
                continue
            want = W[i, k] * equity; have = l.shares * p[k]; diff = want - have
            if abs(diff) > costs.rebalance_band * equity and abs(diff) > 1e-9:
                orders.append((diff, k))
        for diff, k in sorted(orders):                       # 매도(음수) 먼저
            l = lots[k]; pk = p[k]
            if diff < 0:
                notional = min(-diff, l.shares * pk); sold = notional / pk
                realized += notional - sold * l.avg
                l.shares -= sold
                proceeds = notional * (1.0 - costs.sell_fee); cash += proceeds; fees += notional - proceeds; trades += 1
                if l.shares < 1e-12:
                    l.shares, l.avg = 0.0, 0.0
            else:
                cost = diff * (1.0 + costs.buy_fee)
                if cost > cash:
                    diff = cash / (1.0 + costs.buy_fee); cost = cash
                if diff <= 0:
                    continue
                new = diff / pk
                l.avg = (l.shares * l.avg + diff) / (l.shares + new); l.shares += new
                cash -= cost; fees += cost - diff; trades += 1
        equity = cash + sum(l.shares * (p[k] if not np.isnan(p[k]) else 0.0) for k, l in enumerate(lots))
        if i == len(px) - 1 or years[i + 1] != years[i]:
            due = max(0.0, realized - costs.tax_deduction) * costs.tax_rate
            cash -= due; tax += due; realized = 0.0
            equity = cash + sum(l.shares * (p[k] if not np.isnan(p[k]) else 0.0) for k, l in enumerate(lots))
        eq[i] = equity
        ex[i] = (equity - cash) / equity if equity > 0 else 0.0
    # 청산: 전량 매도 + 세금 (연말 정산 직후라 realized 는 0 에서 시작)
    p = P[-1]; notional = 0.0; gain = 0.0
    for k, l in enumerate(lots):
        if l.shares > 0 and not np.isnan(p[k]):
            notional += l.shares * p[k]; gain += l.shares * (p[k] - l.avg)
    final = cash + notional * (1.0 - costs.sell_fee) - max(0.0, realized + gain - costs.tax_deduction) * costs.tax_rate
    equity_s = pd.Series(eq, index=px.index, name=name or "equity")
    res = LabResult(name=name, equity=equity_s, exposure=pd.Series(ex, index=px.index), target=w.sum(axis=1),
                    trades=trades, fees=fees, tax=tax, final_after_liquidation=final)
    res.metrics = compute_risk_metrics(equity_s)
    return res
