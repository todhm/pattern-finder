"""여러 종목 동시 '전일 급락 매수' 시뮬레이터 — 한 계좌, 공유 현금.

단일 종목 :mod:`dip_buy_strategy`의 확장:

- 매일 유니버스 전 종목의 전날 등락률을 검사해, −drop% 이하인
  종목마다 계좌 총액의 1/split을 다음 날 시가 매수. 같은 날 신호가
  겹치면 낙폭 깊은 순으로 현금 배정 (3배 ETF는 폭락이 동행하므로
  현금 제약이 실제로 자주 걸린다).
- 종목별 평단·지정가 익절·스택(최대 split회)은 독립. 익절로 돌아온
  현금은 즉시 다른 종목 신호에 재사용 — 자본 가동률이 단일 종목보다
  높아지는 것이 이 변형의 가설이다.
- 상장일이 늦은 종목(KORU 2013 등)은 데이터 시작 후부터 참여.

수수료·양도세(연 정산 + 최종 청산)는 다른 전략과 동일.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy.adapters.band_rebalance_strategy import _summarize
from strategy.domain.models import (
    DipBuyCycle,
    EquityPoint,
    LiquidationSummary,
    MultiDipBuyConfig,
    MultiDipBuyResult,
)


class _Position:
    __slots__ = (
        "shares", "notional_cost", "tax_cost", "stack",
        "cycle_start", "invested",
    )

    def __init__(self):
        self.shares = 0.0
        self.notional_cost = 0.0  # 평단용 (수수료 제외)
        self.tax_cost = 0.0  # 세무 취득원가 (수수료 포함)
        self.stack = 0
        self.cycle_start = None
        self.invested = 0.0

    @property
    def avg(self) -> float:
        return self.notional_cost / self.shares if self.shares > 0 else 0.0


class MultiDipBuyStrategy:
    """공유 현금 멀티 종목 급락 매수."""

    def execute(
        self, dailies: dict[str, pd.DataFrame], config: MultiDipBuyConfig
    ) -> MultiDipBuyResult:
        tickers = [t for t in config.tickers if t in dailies]
        if not tickers:
            raise ValueError("no ticker data provided")

        # 합집합 달력 위에 종목별 O/H/C/전일수익률 행렬 구성.
        frames = {
            t: dailies[t].dropna(subset=["Open", "High", "Close"])
            for t in tickers
        }
        index = None
        for df in frames.values():
            index = df.index if index is None else index.union(df.index)
        n = len(index)
        if n < 3:
            raise ValueError("need at least 3 daily bars")
        O, H, C, R = {}, {}, {}, {}
        for t, df in frames.items():
            aligned = df.reindex(index)
            O[t] = aligned["Open"].to_numpy(float)
            H[t] = aligned["High"].to_numpy(float)
            C[t] = aligned["Close"].to_numpy(float)
            R[t] = aligned["Close"].pct_change().to_numpy(float)
        last_close = {t: np.nan for t in tickers}

        fee = config.fee_schedule
        sell_rate = fee.sell_commission_pct + fee.sec_fee_pct
        cash = config.initial_capital
        pos = {t: _Position() for t in tickers}
        realized_ytd = 0.0
        realized_total = 0.0
        total_fees = 0.0
        total_tax = 0.0
        cycles: list[DipBuyCycle] = []
        equity: list[float] = []
        prev_year = index[0].year

        def total_value() -> float:
            v = cash
            for t in tickers:
                if pos[t].shares > 0 and not np.isnan(last_close[t]):
                    v += pos[t].shares * last_close[t]
            return v

        def sell_all(t: str, price: float, ts, outcome: str):
            nonlocal cash, realized_ytd, realized_total, total_fees
            p = pos[t]
            notional = p.shares * price
            f = notional * sell_rate
            gain = (notional - f) - p.tax_cost
            realized_ytd += gain
            realized_total += gain
            cash += notional - f
            total_fees += f
            cycles.append(
                DipBuyCycle(
                    ticker=t,
                    start=p.cycle_start,
                    end=ts.date(),
                    holding_days=int((ts.date() - p.cycle_start).days),
                    n_buys=p.stack,
                    avg_price=p.avg,
                    exit_price=price,
                    invested=p.invested,
                    pnl=(notional - f) - p.invested,
                    outcome=outcome,
                )
            )
            pos[t] = _Position()

        for i in range(n):
            ts = index[i]
            for t in tickers:
                if not np.isnan(C[t][i]):
                    last_close[t] = C[t][i]

            # 연초 양도세 정산 (부족하면 가장 큰 포지션부터 매도).
            if ts.year != prev_year:
                tax = max(0.0, realized_ytd - config.tax_deduction)
                tax *= config.capital_gains_tax_pct
                realized_ytd = 0.0
                if tax > 0:
                    while cash < tax:
                        held = [
                            t for t in tickers
                            if pos[t].shares > 0 and not np.isnan(last_close[t])
                        ]
                        if not held:
                            break
                        big = max(held, key=lambda t: pos[t].shares * last_close[t])
                        p = pos[big]
                        price = last_close[big]
                        need = (tax - cash) / (1.0 - sell_rate)
                        qty = min(need / price, p.shares)
                        notional = qty * price
                        f = notional * sell_rate
                        portion = qty / p.shares
                        gain = (notional - f) - p.tax_cost * portion
                        realized_ytd += gain
                        realized_total += gain
                        p.notional_cost *= 1.0 - portion
                        p.tax_cost -= p.tax_cost * portion
                        p.shares -= qty
                        cash += notional - f
                        total_fees += f
                        if p.shares <= 1e-12:
                            pos[big] = _Position()
                    cash -= tax
                    total_tax += tax
                prev_year = ts.year

            sold_today: set[str] = set()
            # --- 1) 종목별 지정가 익절 ---
            for t in tickers:
                p = pos[t]
                if p.shares <= 0 or np.isnan(O[t][i]):
                    continue
                target_price = p.avg * (1.0 + config.target_pct)
                fill = None
                if O[t][i] >= target_price:
                    fill = O[t][i]
                elif H[t][i] >= target_price:
                    fill = target_price
                if fill is not None:
                    sell_all(t, fill, ts, "profit")
                    sold_today.add(t)

            # --- 2) 진입: 낙폭 깊은 순으로 현금 배정 ---
            signals = [
                (R[t][i - 1], t)
                for t in tickers
                if i >= 1
                and not np.isnan(R[t][i - 1])
                and R[t][i - 1] <= -config.drop_pct
                and not np.isnan(O[t][i])
                and t not in sold_today
                and pos[t].stack < config.split
            ]
            for _, t in sorted(signals):
                if cash <= 1e-6:
                    break
                amount = min(total_value() / config.split, cash)
                if amount <= 1e-6:
                    break
                o = O[t][i]
                notional = amount / (1.0 + fee.buy_commission_pct)
                p = pos[t]
                if p.cycle_start is None:
                    p.cycle_start = ts.date()
                p.shares += notional / o
                p.notional_cost += notional
                p.tax_cost += amount
                p.stack += 1
                p.invested += amount
                total_fees += amount - notional
                cash -= amount

            equity.append(total_value())

        # 진행 중 사이클 기록.
        for t in tickers:
            p = pos[t]
            if p.shares > 0 and p.cycle_start is not None:
                cycles.append(
                    DipBuyCycle(
                        ticker=t,
                        start=p.cycle_start,
                        end=None,
                        holding_days=int(
                            (index[-1].date() - p.cycle_start).days
                        ),
                        n_buys=p.stack,
                        avg_price=p.avg,
                        exit_price=None,
                        invested=p.invested,
                        pnl=p.shares * last_close[t] - p.invested
                        if not np.isnan(last_close[t])
                        else 0.0,
                        outcome="open",
                    )
                )

        values = pd.Series(equity, index=index)
        pre_tax = float(values.iloc[-1])
        sale_fee = 0.0
        final_gain = 0.0
        for t in tickers:
            p = pos[t]
            if p.shares > 0 and not np.isnan(last_close[t]):
                notional = p.shares * last_close[t]
                f = notional * sell_rate
                sale_fee += f
                final_gain += (notional - f) - p.tax_cost
        taxable = max(0.0, realized_ytd + final_gain - config.tax_deduction)
        final_tax = taxable * config.capital_gains_tax_pct
        liquidation = LiquidationSummary(
            final_value_pre_tax=pre_tax,
            final_value_after_tax=pre_tax - sale_fee - final_tax,
            final_tax=final_tax,
            total_fees=total_fees + sale_fee,
            total_tax=total_tax + final_tax,
            realized_gain_total=realized_total + final_gain,
        )

        # 벤치마크: 각 종목 상장 시점부터 1/k 동일가중 B&H 세후.
        capital = config.initial_capital
        k = len(tickers)
        bench_after = 0.0
        for t in tickers:
            closes = frames[t]["Close"]
            sleeve = capital / k
            final = float(closes.iloc[-1] / closes.iloc[0]) * (
                sleeve / (1.0 + fee.buy_commission_pct)
            )
            f = final * sell_rate
            gain = max(
                0.0, final - f - sleeve - config.tax_deduction / k
            )
            bench_after += final - f - gain * config.capital_gains_tax_pct

        name = (
            f"멀티 급락매수 [{'+'.join(tickers)}] −{config.drop_pct:.0%} → "
            f"1/{config.split} → +{config.target_pct:.1%}"
        )
        return MultiDipBuyResult(
            config=config,
            summary=_summarize(name, values, capital),
            liquidation=liquidation,
            cycles=cycles,
            equity_curve=[
                EquityPoint(date=ts.date(), equity=float(v))
                for ts, v in values.items()
            ],
            benchmark_after_tax=bench_after,
        )
