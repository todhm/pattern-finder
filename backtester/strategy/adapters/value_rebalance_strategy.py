"""라오어 '밸류 리밸런싱(VR)' 전략 — TQQQ 거치식.

책/공개 자료 기준 규칙:

- 초기 자본을 주식 : Pool(현금) = ``stock_ratio`` : 나머지로 분할,
  **V(밸류패스) = 초기 주식 평가금**으로 시작.
- 2주(기본 10거래일)마다 **V += Pool / G** (거치식, G 기본 10).
  Pool이 두둑하면 목표를 빨리 올리고, 소진되면 천천히 올리는
  자기조절 구조.
- 평가금 E가 밴드 상단 V×(1+15%)를 넘으면 초과분(E−V)을 매도해
  Pool에 적립, 하단 V×(1−15%) 아래로 내려가면 부족분(V−E)을
  Pool 한도 내에서 매수한다.
- ``check_daily=True``(기본)면 매일 밴드를 검사한다 — 책의 매수표·
  매도표 LOC 주문 방식의 근사. False면 2주 시점에만 검사.
- ``advanced_formula``(실력공식 근사): V 증가분에 √(E/V)를 곱해
  하락장에서 밸류패스 상승을 늦춘다. 원본 공식은 서적 전용이라
  공개된 설명 기반의 근사임.

수수료(토스증권)·양도소득세(연 단위 정산 + 최종 청산)는 TQQQ P2P
전략과 동일한 방식으로 반영한다. `execute`는 정렬된 종가 Series만
받는 순수 어댑터 — 데이터 fetch는 composition root의 몫.
"""

from __future__ import annotations

import math

import pandas as pd

from strategy.adapters.band_rebalance_strategy import _summarize
from strategy.domain.models import (
    EquityPoint,
    LiquidationSummary,
    PortfolioSummary,
    ValueRebalanceConfig,
    ValueRebalanceEvent,
    ValueRebalancePoint,
    ValueRebalanceResult,
)


class ValueRebalanceStrategy:
    """밸류패스 추종 리밸런싱 시뮬레이터."""

    def execute(
        self, close: pd.Series, config: ValueRebalanceConfig
    ) -> ValueRebalanceResult:
        prices = close.dropna()
        if len(prices) < 2:
            raise ValueError(
                f"Need at least 2 bars to run the backtest (got {len(prices)})"
            )

        fee = config.fee_schedule
        sell_rate = fee.sell_commission_pct + fee.sec_fee_pct
        band = config.band_pct

        shares = 0.0
        avg_cost = 0.0
        pool = 0.0
        realized_ytd = 0.0
        realized_total = 0.0
        total_fees = 0.0
        total_tax = 0.0
        total_interest = 0.0
        daily_pool_rate = config.pool_annual_rate / 252.0

        def buy(amount: float, price: float) -> float:
            """Pool에서 ``amount``(수수료 포함)를 꺼내 매수. 노셔널 반환."""
            nonlocal shares, avg_cost, pool, total_fees
            if amount <= 0:
                return 0.0
            notional = amount / (1.0 + fee.buy_commission_pct)
            prev_cost = shares * avg_cost
            shares += notional / price
            avg_cost = (prev_cost + amount) / shares
            pool -= amount
            total_fees += amount - notional
            return notional

        def sell(notional: float, price: float) -> float:
            """평가액 ``notional`` 매도 → Pool 적립. 노셔널 반환."""
            nonlocal shares, pool, realized_ytd, realized_total, total_fees
            notional = min(notional, shares * price)
            if notional <= 0:
                return 0.0
            sold = notional / price
            f = notional * sell_rate
            gain = (notional - f) - sold * avg_cost
            realized_ytd += gain
            realized_total += gain
            shares -= sold
            pool += notional - f
            total_fees += f
            return notional

        def settle_tax(price: float) -> float:
            nonlocal realized_ytd, pool, total_tax
            tax = max(0.0, realized_ytd - config.tax_deduction)
            tax *= config.capital_gains_tax_pct
            realized_ytd = 0.0
            if tax <= 0:
                return 0.0
            if pool < tax and shares > 0:
                sell((tax - pool) / (1.0 - sell_rate), price)
            pool -= tax
            total_tax += tax
            return tax

        # --- Day 0: 분할 매수, V = 초기 주식 평가금 ---
        p0 = float(prices.iloc[0])
        pool = config.initial_capital
        buy(config.initial_capital * config.stock_ratio, p0)
        value_path = shares * p0

        curve: list[ValueRebalancePoint] = []
        events: list[ValueRebalanceEvent] = []
        prev_ts = prices.index[0]
        bar_count = 0

        for ts in prices.index:
            p = float(prices.loc[ts])

            if ts != prices.index[0]:
                bar_count += 1

                # Pool 이자 (P2P 채권 운용 가정 — 일할 누적).
                if daily_pool_rate > 0 and pool > 0:
                    interest = pool * daily_pool_rate
                    pool += interest
                    total_interest += interest

                # 연초: 전년도 실현손익 양도세 정산.
                if ts.year != prev_ts.year:
                    tax_paid = settle_tax(p)
                    if tax_paid > 0:
                        events.append(
                            ValueRebalanceEvent(
                                date=ts.date(),
                                kind="tax",
                                traded=-tax_paid,
                                value_path=value_path,
                                stock_value_after=shares * p,
                                pool_after=pool,
                            )
                        )

                # 사이클: V += Pool / G (실력공식은 √(E/V) 보정).
                if bar_count % config.cycle_days == 0:
                    step = pool / config.gradient
                    if config.advanced_formula and value_path > 0:
                        step *= math.sqrt(max(shares * p, 0.0) / value_path)
                    value_path += step

                # 밴드 검사 (daily 또는 사이클 시점만). 체결 목표는
                # 밴드 가장자리(기본 — 매수표·매도표 LOC 분할 체결
                # 등가) 또는 V(공격적).
                if config.check_daily or bar_count % config.cycle_days == 0:
                    e_val = shares * p
                    upper = value_path * (1.0 + band)
                    lower = value_path * (1.0 - band)
                    if e_val > upper:
                        target = upper if config.rebalance_to_edge else value_path
                        traded = -sell(e_val - target, p)
                        events.append(
                            ValueRebalanceEvent(
                                date=ts.date(),
                                kind="sell",
                                traded=traded,
                                value_path=value_path,
                                stock_value_after=shares * p,
                                pool_after=pool,
                            )
                        )
                    elif e_val < lower and pool > 1e-9:
                        target = lower if config.rebalance_to_edge else value_path
                        traded = buy(min(target - e_val, pool), p)
                        events.append(
                            ValueRebalanceEvent(
                                date=ts.date(),
                                kind="buy",
                                traded=traded,
                                value_path=value_path,
                                stock_value_after=shares * p,
                                pool_after=pool,
                            )
                        )

            curve.append(
                ValueRebalancePoint(
                    date=ts.date(),
                    total=shares * p + pool,
                    stock_value=shares * p,
                    pool=pool,
                    value_path=value_path,
                )
            )
            prev_ts = ts

        # --- 최종 청산 (마지막 봉 종가) ---
        p_last = float(prices.iloc[-1])
        pre_tax = curve[-1].total
        sale_notional = shares * p_last
        sale_fee = sale_notional * sell_rate
        final_gain = (sale_notional - sale_fee) - shares * avg_cost
        taxable = max(0.0, realized_ytd + final_gain - config.tax_deduction)
        final_tax = taxable * config.capital_gains_tax_pct
        liquidation = LiquidationSummary(
            final_value_pre_tax=pre_tax,
            final_value_after_tax=pre_tax - sale_fee - final_tax,
            final_tax=final_tax,
            total_interest=total_interest,
            total_fees=total_fees + sale_fee,
            total_tax=total_tax + final_tax,
            realized_gain_total=realized_total + final_gain,
        )

        strategy_values = pd.Series(
            [pt.total for pt in curve], index=prices.index
        )
        formula = "실력공식(√보정)" if config.advanced_formula else "기본공식"
        name = (
            f"라오어 VR {formula} — {config.ticker} "
            f"{config.stock_ratio:.0%}/Pool, G={config.gradient:.0f}, "
            f"±{band:.0%}"
        )
        if config.pool_annual_rate > 0:
            name += f" + Pool P2P {config.pool_annual_rate:.0%}"

        # --- 벤치마크: TQQQ 100% / 초기 비율 방치 ---
        capital = config.initial_capital
        norm = prices / p0
        tqqq_bh = norm * (capital / (1.0 + fee.buy_commission_pct))
        untouched = tqqq_bh * config.stock_ratio + capital * (
            1.0 - config.stock_ratio
        )
        bench_defs = {
            f"{config.ticker} 100%": tqqq_bh,
            f"{config.stock_ratio:.0%}:{1 - config.stock_ratio:.0%} 방치": untouched,
        }
        benchmarks: list[PortfolioSummary] = []
        benchmark_curves: dict[str, list[EquityPoint]] = {}
        benchmark_after_tax: dict[str, float] = {}

        def _after_tax_bh(stock_value: float, cost: float, rest: float) -> float:
            f = stock_value * sell_rate
            gain = max(
                0.0, stock_value - f - cost - config.tax_deduction
            )
            return stock_value - f - gain * config.capital_gains_tax_pct + rest

        for bench_name, values in bench_defs.items():
            benchmarks.append(_summarize(bench_name, values, capital))
            benchmark_curves[bench_name] = [
                EquityPoint(date=ts.date(), equity=float(v))
                for ts, v in values.items()
            ]
        benchmark_after_tax[f"{config.ticker} 100%"] = _after_tax_bh(
            float(tqqq_bh.iloc[-1]), capital, 0.0
        )
        benchmark_after_tax[
            f"{config.stock_ratio:.0%}:{1 - config.stock_ratio:.0%} 방치"
        ] = _after_tax_bh(
            float(tqqq_bh.iloc[-1]) * config.stock_ratio,
            capital * config.stock_ratio,
            capital * (1.0 - config.stock_ratio),
        )

        return ValueRebalanceResult(
            config=config,
            curve=curve,
            events=events,
            summary=_summarize(name, strategy_values, capital),
            liquidation=liquidation,
            benchmarks=benchmarks,
            benchmark_curves=benchmark_curves,
            benchmark_after_tax=benchmark_after_tax,
        )
