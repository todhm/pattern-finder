"""'공포에 사는' 역발상 사다리 전략 시뮬레이터.

거시 레짐 필터(위기 회피)의 정반대 — 구루 방법론 구현:

- 버핏: "남들이 두려워할 때 탐욕을" — 평시에 현금 예비대를 쌓고
  폭락에 투입.
- 하워드 막스: 사이클 극단(비관 극대) 포지셔닝 — 낙폭이 깊을수록
  투입 비중을 키운다 (사다리).
- "200MA 아래 = 기회": ``trigger_mode="ma"``면 기준선이 고점이 아닌
  200일선이라, 이평선 이탈 깊이에 비례해 매수한다.

규칙 상세는 :class:`FearLadderConfig` 참조. 체결은 당일 종가,
회복 리밸런싱의 매도는 실현손익으로 과세(연 정산 + 최종 청산).
`execute`는 정렬된 종가 Series(+선택적 VIX)만 받는 순수 어댑터.
"""

from __future__ import annotations

import pandas as pd

from strategy.adapters.band_rebalance_strategy import _summarize
from strategy.domain.models import (
    EquityPoint,
    FearCycle,
    FearLadderConfig,
    FearLadderEvent,
    FearLadderResult,
    LiquidationSummary,
    PortfolioSummary,
)


class FearLadderStrategy:
    """드로다운/이평 이탈 사다리 매수 + 회복 익절 시뮬레이터."""

    def execute(
        self,
        close: pd.Series,
        config: FearLadderConfig,
        vix: pd.Series | None = None,
    ) -> FearLadderResult:
        prices = close.dropna()
        if len(prices) < 2:
            raise ValueError(
                f"Need at least 2 bars to run the backtest (got {len(prices)})"
            )
        if len(config.levels) != len(config.deploy_fractions):
            raise ValueError("levels와 deploy_fractions 길이가 달라야 함")

        fee = config.fee_schedule
        sell_rate = fee.sell_commission_pct + fee.sec_fee_pct
        daily_rate = config.cash_annual_rate / 252.0
        use_ma = config.trigger_mode == "ma"
        ma = prices.rolling(config.ma_days).mean() if use_ma else None
        vix_smooth = None
        if config.vix_confirm and vix is not None:
            vix_smooth = (
                vix.rolling(21).mean().reindex(prices.index, method="ffill")
            )

        shares = 0.0
        avg_cost = 0.0  # 주당 취득원가 (수수료 자본화)
        cash = config.initial_capital
        realized_ytd = 0.0
        realized_total = 0.0
        total_fees = 0.0
        total_tax = 0.0
        total_interest = 0.0

        levels_hit: set[int] = set()
        in_cycle = False
        cycle_start = None
        cycle_invested = 0.0
        cycle_min_dd = 0.0
        recovery_streak = 0
        ath = float(prices.iloc[0])

        events: list[FearLadderEvent] = []
        cycles: list[FearCycle] = []
        equity: list[float] = []
        prev_year = prices.index[0].year

        def record(ts, kind, price, notional, level=0, dd=0.0):
            total = shares * price + cash
            events.append(
                FearLadderEvent(
                    date=ts.date(),
                    kind=kind,
                    level=level,
                    drawdown=dd,
                    price=price,
                    notional=notional,
                    stock_value_after=shares * price,
                    cash_after=cash,
                    stock_weight_after=shares * price / total if total > 0 else 0.0,
                )
            )

        def buy(ts, kind, amount, price, level=0, dd=0.0):
            nonlocal shares, avg_cost, cash, total_fees, cycle_invested
            amount = min(amount, cash)
            if amount <= 0 or price <= 0:
                return
            notional = amount / (1.0 + fee.buy_commission_pct)
            prev = shares * avg_cost
            shares += notional / price
            avg_cost = (prev + amount) / shares
            cash -= amount
            total_fees += amount - notional
            if kind == "fear_buy":
                cycle_invested += amount
            record(ts, kind, price, notional, level, dd)

        def sell(notional, price):
            nonlocal shares, cash, realized_ytd, realized_total, total_fees
            notional = min(notional, shares * price)
            if notional <= 0:
                return 0.0
            qty = notional / price
            f = notional * sell_rate
            gain = (notional - f) - qty * avg_cost
            realized_ytd += gain
            realized_total += gain
            shares -= qty
            cash += notional - f
            total_fees += f
            return notional

        # --- Day 0: 평시 비중 매수 ---
        p0 = float(prices.iloc[0])
        buy(prices.index[0], "base_buy", config.initial_capital * config.base_stock_weight, p0)

        for i, ts in enumerate(prices.index):
            p = float(prices.loc[ts])

            if i > 0:
                # 현금 이자 (P2P 등).
                if daily_rate > 0 and cash > 0:
                    interest = cash * daily_rate
                    cash += interest
                    total_interest += interest

                # 연초 양도세 정산.
                if ts.year != prev_year:
                    tax = max(0.0, realized_ytd - config.tax_deduction)
                    tax *= config.capital_gains_tax_pct
                    realized_ytd = 0.0
                    if tax > 0:
                        if cash < tax and shares > 0:
                            sell((tax - cash) / (1.0 - sell_rate), p)
                        cash -= tax
                        total_tax += tax
                        record(ts, "tax", p, -tax)
                    prev_year = ts.year

                # --- 기준선/낙폭 ---
                if use_ma:
                    ref = float(ma.loc[ts]) if not pd.isna(ma.loc[ts]) else None
                else:
                    ref = ath
                dd = p / ref - 1.0 if ref else 0.0
                if in_cycle:
                    cycle_min_dd = min(cycle_min_dd, dd)

                # --- 회복 판정 ---
                recovered = False
                if in_cycle:
                    if use_ma:
                        if ref is not None and p > ref:
                            recovery_streak += 1
                            if recovery_streak > config.recovery_confirm_days:
                                recovered = True
                        else:
                            recovery_streak = 0
                    elif p > ath:  # 신고점 = 완전 회복
                        recovered = True
                if recovered:
                    total = shares * p + cash
                    target = total * config.base_stock_weight
                    harvested = 0.0
                    if shares * p > target:
                        harvested = sell(shares * p - target, p)
                    record(ts, "recovery", p, -harvested)
                    cycles.append(
                        FearCycle(
                            start=cycle_start,
                            end=ts.date(),
                            min_drawdown=cycle_min_dd,
                            levels_hit=len(levels_hit),
                            invested=cycle_invested,
                            harvested=harvested,
                            outcome="recovered",
                        )
                    )
                    in_cycle = False
                    levels_hit.clear()
                    cycle_invested = 0.0
                    cycle_min_dd = 0.0
                    recovery_streak = 0

                # --- 사다리 발동 ---
                fear_ok = True
                if vix_smooth is not None:
                    v = vix_smooth.loc[ts]
                    fear_ok = (not pd.isna(v)) and v >= config.vix_threshold
                if ref and fear_ok:
                    for j, lvl in enumerate(config.levels):
                        if j in levels_hit or dd > lvl:
                            continue
                        if not in_cycle:
                            in_cycle = True
                            cycle_start = ts.date()
                            cycle_min_dd = dd
                        levels_hit.add(j)
                        buy(
                            ts, "fear_buy",
                            cash * config.deploy_fractions[j], p,
                            level=j + 1, dd=dd,
                        )

                if not use_ma:
                    ath = max(ath, p)

            equity.append(shares * p + cash)

        if in_cycle:
            cycles.append(
                FearCycle(
                    start=cycle_start,
                    end=None,
                    min_drawdown=cycle_min_dd,
                    levels_hit=len(levels_hit),
                    invested=cycle_invested,
                    harvested=0.0,
                    outcome="open",
                )
            )

        values = pd.Series(equity, index=prices.index)
        p_last = float(prices.iloc[-1])
        pre_tax = float(values.iloc[-1])
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

        # --- 벤치마크 ---
        capital = config.initial_capital
        norm = prices / p0
        bh = norm * (capital / (1.0 + fee.buy_commission_pct))
        w = config.base_stock_weight
        untouched = bh * w + capital * (1.0 - w)
        bench_defs = {
            f"{config.ticker} 100%": bh,
            f"{w:.0%} 방치 (사다리 없음)": untouched,
        }
        benchmarks: list[PortfolioSummary] = []
        benchmark_curves: dict[str, list[EquityPoint]] = {}
        benchmark_after_tax: dict[str, float] = {}

        def _bh_after_tax(v, cost, rest):
            f = v * sell_rate
            g = max(0.0, v - f - cost - config.tax_deduction)
            return v - f - g * config.capital_gains_tax_pct + rest

        for name, series in bench_defs.items():
            benchmarks.append(_summarize(name, series, capital))
            benchmark_curves[name] = [
                EquityPoint(date=t.date(), equity=float(v))
                for t, v in series.items()
            ]
        benchmark_after_tax[f"{config.ticker} 100%"] = _bh_after_tax(
            float(bh.iloc[-1]), capital, 0.0
        )
        benchmark_after_tax[f"{w:.0%} 방치 (사다리 없음)"] = _bh_after_tax(
            float(bh.iloc[-1]) * w, capital * w, capital * (1.0 - w)
        )

        mode_label = (
            f"{config.ma_days}MA 이탈" if use_ma else "고점 드로다운"
        )
        name = (
            f"공포 사다리 ({mode_label} "
            f"{'/'.join(f'{abs(v):.0%}' for v in config.levels)}) — "
            f"{config.ticker} {w:.0%}+현금"
        )
        if config.cash_annual_rate > 0:
            name += f" P2P {config.cash_annual_rate:.0%}"
        return FearLadderResult(
            config=config,
            summary=_summarize(name, values, capital),
            liquidation=liquidation,
            events=events,
            cycles=cycles,
            equity_curve=[
                EquityPoint(date=t.date(), equity=float(v))
                for t, v in values.items()
            ],
            benchmarks=benchmarks,
            benchmark_curves=benchmark_curves,
            benchmark_after_tax=benchmark_after_tax,
        )
