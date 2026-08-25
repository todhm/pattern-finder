"""TQQQ + P2P 채권(연 9%, 월 이자, 6개월 만기) 50:50 전략.

P2P 투자법인 운영 시나리오:

- 자본의 50%는 TQQQ, 50%는 P2P 채권 사다리에 투자.
- 채권은 매월 이자(연리/12)를 지급하고 만기(기본 12개월)에 원금 상환.
- 매월 첫 거래일에 들어온 이자 + 만기 원금을 **목표 비중에 모자란
  쪽부터** 투자해 50:50을 유지한다 (현금흐름 리밸런싱). 옵션으로
  TQQQ 초과분 매도까지 허용하는 완전 리밸런싱 모드 지원.
- TQQQ 매매엔 토스증권 수수료(:class:`TossFeeSchedule`), 실현 차익엔
  연 단위 양도소득세(기본 22%, 기본공제 차감)를 적용한다.

`execute`는 정렬된 TQQQ 종가 Series만 받는 순수 어댑터 — 데이터
fetch는 composition root(Streamlit 페이지)의 몫. 체결은 당일 종가.
채권은 시장가 없이 액면(원금)으로 평가한다.
"""

from __future__ import annotations

import pandas as pd

from strategy.adapters.band_rebalance_strategy import _summarize
from strategy.domain.models import (
    EquityPoint,
    PortfolioSummary,
    TqqqP2pConfig,
    TqqqP2pEvent,
    TqqqP2pLiquidation,
    TqqqP2pPoint,
    TqqqP2pResult,
)


class TqqqP2pStrategy:
    """TQQQ + P2P 채권 사다리 시뮬레이터."""

    def execute(self, close: pd.Series, config: TqqqP2pConfig) -> TqqqP2pResult:
        prices = close.dropna()
        if len(prices) < 2:
            raise ValueError(
                f"Need at least 2 bars to run the backtest (got {len(prices)})"
            )

        fee = config.fee_schedule
        mrate = config.bond_annual_rate / 12.0
        w = config.tqqq_weight
        sell_rate = fee.sell_commission_pct + fee.sec_fee_pct

        # --- 상태 ---
        shares = 0.0
        avg_cost = 0.0  # 주당 취득원가 (매수 수수료 자본화)
        bonds: list[list[float]] = []  # [원금, 남은 개월]
        cash = 0.0
        realized_ytd = 0.0
        total_interest = 0.0
        total_fees = 0.0
        total_tax = 0.0
        realized_total = 0.0

        def buy(amount: float, price: float) -> float:
            """현금 ``amount``를 전부 써서 매수 (수수료 포함). 노셔널 반환."""
            nonlocal shares, avg_cost, cash, total_fees
            if amount <= 0:
                return 0.0
            notional = amount / (1.0 + fee.buy_commission_pct)
            prev_cost = shares * avg_cost
            shares += notional / price
            avg_cost = (prev_cost + amount) / shares
            cash -= amount
            total_fees += amount - notional
            return notional

        def sell(notional: float, price: float) -> float:
            """평가액 ``notional``만큼 매도. 실현손익 누적. 노셔널 반환."""
            nonlocal shares, cash, realized_ytd, realized_total, total_fees
            notional = min(notional, shares * price)
            if notional <= 0:
                return 0.0
            sold = notional / price
            f = notional * sell_rate
            gain = (notional - f) - sold * avg_cost
            realized_ytd += gain
            realized_total += gain
            shares -= sold
            cash += notional - f
            total_fees += f
            return notional

        def settle_tax(price: float) -> float:
            """연간 실현손익에 양도세 부과. 현금 부족 시 TQQQ 매도로 충당."""
            nonlocal realized_ytd, cash, total_tax
            tax = max(0.0, realized_ytd - config.tax_deduction)
            tax *= config.capital_gains_tax_pct
            realized_ytd = 0.0
            if tax <= 0:
                return 0.0
            if cash < tax and shares > 0:
                shortfall = (tax - cash) / (1.0 - sell_rate)
                sell(shortfall, price)
            cash -= tax
            total_tax += tax
            return tax

        # --- Day 0: 50:50 분할 매수 ---
        p0 = float(prices.iloc[0])
        cash = config.initial_capital
        buy(config.initial_capital * w, p0)
        if cash > 0:
            bonds.append([cash, float(config.bond_maturity_months)])
            cash = 0.0

        curve: list[TqqqP2pPoint] = []
        events: list[TqqqP2pEvent] = []
        boundaries: list[pd.Timestamp] = []
        prev_ts = prices.index[0]

        for ts in prices.index:
            p = float(prices.loc[ts])

            is_boundary = ts != prices.index[0] and (
                ts.month != prev_ts.month or ts.year != prev_ts.year
            )
            if is_boundary:
                boundaries.append(ts)
                # 1) 연초: 전년도 실현손익 과세.
                tax_paid = (
                    settle_tax(p) if ts.year != prev_ts.year else 0.0
                )
                # 2) 기존 채권 이자 지급 + 만기 상환.
                interest = sum(b[0] for b in bonds) * mrate
                cash += interest
                total_interest += interest
                for b in bonds:
                    b[1] -= 1
                matured = sum(b[0] for b in bonds if b[1] <= 0)
                cash += matured
                bonds = [b for b in bonds if b[1] > 0]

                # 3) 리밸런싱: 모자란 쪽부터 현금 투입.
                tqqq_val = shares * p
                bond_val = sum(b[0] for b in bonds)
                total = tqqq_val + bond_val + cash
                target = total * w
                traded = 0.0
                if config.allow_sell_rebalance and tqqq_val > target:
                    traded = -sell(tqqq_val - target, p)
                    tqqq_val = shares * p
                deficit = max(0.0, target - tqqq_val)
                buy_amount = min(cash, deficit)
                if buy_amount > 0:
                    traded += buy(buy_amount, p)
                invested = 0.0
                if cash > 1e-9:
                    invested = cash
                    bonds.append([cash, float(config.bond_maturity_months)])
                    cash = 0.0

                tqqq_val = shares * p
                bond_val = sum(b[0] for b in bonds)
                total = tqqq_val + bond_val + cash
                events.append(
                    TqqqP2pEvent(
                        date=ts.date(),
                        interest=interest,
                        matured_principal=matured,
                        tax_paid=tax_paid,
                        tqqq_traded=traded,
                        bond_invested=invested,
                        tqqq_value_after=tqqq_val,
                        bond_value_after=bond_val,
                        cash_after=cash,
                        tqqq_weight_after=tqqq_val / total if total > 0 else 0.0,
                    )
                )

            bond_val = sum(b[0] for b in bonds)
            curve.append(
                TqqqP2pPoint(
                    date=ts.date(),
                    total=shares * p + bond_val + cash,
                    tqqq_value=shares * p,
                    bond_value=bond_val,
                    cash=cash,
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
        after_tax = pre_tax - sale_fee - final_tax
        liquidation = TqqqP2pLiquidation(
            final_value_pre_tax=pre_tax,
            final_value_after_tax=after_tax,
            final_tax=final_tax,
            total_interest=total_interest,
            total_fees=total_fees + sale_fee,
            total_tax=total_tax + final_tax,
            realized_gain_total=realized_total + final_gain,
        )

        strategy_values = pd.Series(
            [pt.total for pt in curve], index=prices.index
        )
        mode = "완전 리밸런싱" if config.allow_sell_rebalance else "현금흐름 리밸런싱"
        name = (
            f"{config.ticker} {w:.0%} + P2P 채권 "
            f"(연 {config.bond_annual_rate:.0%}, {mode})"
        )

        # --- 벤치마크 ---
        capital = config.initial_capital
        norm = prices / p0
        # TQQQ 100% (매수 수수료 반영 buy & hold).
        tqqq_notional = capital / (1.0 + fee.buy_commission_pct)
        tqqq_bh = norm * tqqq_notional
        # P2P 채권 100%: 이자를 매월 새 채권에 재투자 = 월복리.
        factor, factors = 1.0, {}
        for b_ts in boundaries:
            factor *= 1.0 + mrate
            factors[b_ts] = factor
        bond_factor = (
            pd.Series(factors).reindex(prices.index).ffill().fillna(1.0)
        )
        bond_only = bond_factor * capital
        # 50:50 방치: 각자 굴러가고 서로 옮기지 않음.
        untouched = tqqq_bh * w + bond_only * (1.0 - w)

        bench_defs = {
            f"{config.ticker} 100%": tqqq_bh,
            f"P2P 채권 100% (연 {config.bond_annual_rate:.0%} 월복리)": bond_only,
            "50:50 방치 (리밸런싱 없음)": untouched,
        }
        benchmarks: list[PortfolioSummary] = []
        benchmark_curves: dict[str, list[EquityPoint]] = {}
        benchmark_after_tax: dict[str, float] = {}
        for bench_name, values in bench_defs.items():
            benchmarks.append(_summarize(bench_name, values, capital))
            benchmark_curves[bench_name] = [
                EquityPoint(date=ts.date(), equity=float(v))
                for ts, v in values.items()
            ]
        # 벤치마크 세후 청산가: TQQQ 보유분에만 매도 수수료 + 양도세.
        def _after_tax_bh(tqqq_value: float, tqqq_cost: float, rest: float) -> float:
            f = tqqq_value * sell_rate
            gain = max(0.0, tqqq_value - f - tqqq_cost - config.tax_deduction)
            return tqqq_value - f - gain * config.capital_gains_tax_pct + rest

        benchmark_after_tax[f"{config.ticker} 100%"] = _after_tax_bh(
            float(tqqq_bh.iloc[-1]), capital, 0.0
        )
        benchmark_after_tax[
            f"P2P 채권 100% (연 {config.bond_annual_rate:.0%} 월복리)"
        ] = float(bond_only.iloc[-1])
        benchmark_after_tax["50:50 방치 (리밸런싱 없음)"] = _after_tax_bh(
            float(tqqq_bh.iloc[-1]) * w,
            capital * w,
            float(bond_only.iloc[-1]) * (1.0 - w),
        )

        return TqqqP2pResult(
            config=config,
            curve=curve,
            events=events,
            summary=_summarize(name, strategy_values, capital),
            liquidation=liquidation,
            benchmarks=benchmarks,
            benchmark_curves=benchmark_curves,
            benchmark_after_tax=benchmark_after_tax,
        )
