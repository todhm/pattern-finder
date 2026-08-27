"""전일 급락 분할 매수 전략 (무한매수법 변형) 시뮬레이터.

규칙 (:class:`DipBuyConfig` 참조):

- 전날 종가가 그 전날 대비 −drop% 이하로 떨어졌으면 **다음 날 시가**
  에 계좌 총액의 1/split 매수 (현금 한도 내 물타기 스택).
- 보유 중엔 평단 +target% 지정가 익절이 상시 걸려 있다 — 시가 갭이면
  시가, 장중 고가 터치면 목표가 체결. 손절 없음.
- 매도한 날은 신규 매수를 쉰다 (사이클 경계).

일봉 OHLC만 사용 — 짧은 target(1%대)은 일 단위 고가 터치 판정으로
충분하고, 그리드 탐색(수십 조합)을 수 초 안에 돌리기 위함.
"""

from __future__ import annotations

import pandas as pd

from strategy.adapters.band_rebalance_strategy import _summarize
from strategy.domain.models import (
    DipBuyConfig,
    DipBuyCycle,
    DipBuyEvent,
    DipBuyResult,
    EquityPoint,
    LiquidationSummary,
)


class DipBuyStrategy:
    """전일 −drop% → 1/split 매수 → 평단 +target% 익절."""

    def execute(self, daily: pd.DataFrame, config: DipBuyConfig) -> DipBuyResult:
        df = daily.dropna(subset=["Open", "High", "Close"])
        if len(df) < 3:
            raise ValueError(
                f"Need at least 3 daily bars to run the backtest (got {len(df)})"
            )

        fee = config.fee_schedule
        sell_rate = fee.sell_commission_pct + fee.sec_fee_pct
        opens = df["Open"].to_numpy(dtype=float)
        highs = df["High"].to_numpy(dtype=float)
        closes = df["Close"].to_numpy(dtype=float)
        rets = df["Close"].pct_change().to_numpy(dtype=float)
        index = df.index

        cash = config.initial_capital
        shares = 0.0
        notional_cost = 0.0  # 평단용 (수수료 제외)
        tax_cost = 0.0  # 세무 취득원가 (수수료 포함)
        realized_ytd = 0.0
        realized_total = 0.0
        total_fees = 0.0
        total_tax = 0.0

        cycle_start = None
        cycle_buys = 0
        cycle_invested = 0.0
        cycle_no = 0

        cycles: list[DipBuyCycle] = []
        events: list[DipBuyEvent] = []
        equity: list[float] = []
        prev_year = index[0].year

        def record(ts, kind, price, qty, notional, stack_no=0, trigger=0.0):
            avg = notional_cost / shares if shares > 0 else 0.0
            events.append(
                DipBuyEvent(
                    date=ts.date(),
                    kind=kind,
                    price=price,
                    qty=qty,
                    notional=notional,
                    stack_no=stack_no,
                    trigger_ret=trigger,
                    shares_after=shares,
                    avg_price_after=avg,
                    target_price=avg * (1.0 + config.target_pct)
                    if shares > 0
                    else 0.0,
                    cash_after=cash,
                    cycle_no=cycle_no,
                )
            )

        for i in range(len(df)):
            o, h, c = opens[i], highs[i], closes[i]
            ts = index[i]

            # 연초 양도세 정산 (현금 부족 시 시가 매도로 충당).
            if ts.year != prev_year:
                tax = max(0.0, realized_ytd - config.tax_deduction)
                tax *= config.capital_gains_tax_pct
                realized_ytd = 0.0
                if tax > 0:
                    if cash < tax and shares > 0:
                        need = (tax - cash) / (1.0 - sell_rate)
                        qty = min(need / o, shares)
                        proceeds = qty * o * (1.0 - sell_rate)
                        portion = qty / shares
                        gain = proceeds - tax_cost * portion
                        realized_ytd += gain
                        realized_total += gain
                        total_fees += qty * o * sell_rate
                        notional_cost *= 1.0 - portion
                        tax_cost -= tax_cost * portion
                        shares -= qty
                        cash += proceeds
                    cash -= tax
                    total_tax += tax
                    record(ts, "tax", o, 0.0, -tax)
                prev_year = ts.year

            sold_today = False
            # --- 1) 익절 판정 (시가 갭 → 시가, 고가 터치 → 목표가) ---
            if shares > 0:
                avg = notional_cost / shares
                target_price = avg * (1.0 + config.target_pct)
                fill = None
                if o >= target_price:
                    fill = o
                elif h >= target_price:
                    fill = target_price
                if fill is not None:
                    notional = shares * fill
                    f = notional * sell_rate
                    gain = (notional - f) - tax_cost
                    realized_ytd += gain
                    realized_total += gain
                    cash += notional - f
                    total_fees += f
                    cycles.append(
                        DipBuyCycle(
                            start=cycle_start,
                            end=ts.date(),
                            holding_days=int(
                                (ts.date() - cycle_start).days
                            ),
                            n_buys=cycle_buys,
                            avg_price=avg,
                            exit_price=fill,
                            invested=cycle_invested,
                            pnl=(notional - f) - cycle_invested,
                            outcome="profit",
                        )
                    )
                    sold_qty = shares
                    shares = 0.0
                    notional_cost = 0.0
                    tax_cost = 0.0
                    cycle_start = None
                    cycle_buys = 0
                    cycle_invested = 0.0
                    sold_today = True
                    record(ts, "sell", fill, sold_qty, -notional)

            # --- 2) 진입: 전날 일간 수익률 ≤ −drop% → 시가 매수 ---
            if (
                not sold_today
                and i >= 1
                and rets[i - 1] <= -config.drop_pct
                and cash > 1e-6
                and cycle_buys < config.split
            ):
                total_value = shares * o + cash
                amount = min(total_value / config.split, cash)
                if amount > 1e-6:
                    notional = amount / (1.0 + fee.buy_commission_pct)
                    if cycle_start is None:
                        cycle_start = ts.date()
                        cycle_no += 1
                    shares += notional / o
                    notional_cost += notional
                    tax_cost += amount
                    total_fees += amount - notional
                    cash -= amount
                    cycle_buys += 1
                    cycle_invested += amount
                    record(
                        ts, "buy", o, notional / o, notional,
                        stack_no=cycle_buys, trigger=rets[i - 1],
                    )

            equity.append(shares * c + cash)

        # 진행 중 사이클.
        if shares > 0 and cycle_start is not None:
            cycles.append(
                DipBuyCycle(
                    start=cycle_start,
                    end=None,
                    holding_days=int((index[-1].date() - cycle_start).days),
                    n_buys=cycle_buys,
                    avg_price=notional_cost / shares,
                    exit_price=None,
                    invested=cycle_invested,
                    pnl=shares * closes[-1] - cycle_invested,
                    outcome="open",
                )
            )

        values = pd.Series(equity, index=index)
        pre_tax = float(values.iloc[-1])
        sale_notional = shares * closes[-1]
        sale_fee = sale_notional * sell_rate
        final_gain = (sale_notional - sale_fee) - tax_cost
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

        # 벤치마크: 동일 구간 100% B&H 세후.
        capital = config.initial_capital
        bh_final = closes[-1] / closes[0] * (
            capital / (1.0 + fee.buy_commission_pct)
        )
        bh_fee = bh_final * sell_rate
        bh_gain = max(0.0, bh_final - bh_fee - capital - config.tax_deduction)
        bh_after_tax = bh_final - bh_fee - bh_gain * config.capital_gains_tax_pct

        name = (
            f"전일 −{config.drop_pct:.0%} 급락 매수 1/{config.split} → "
            f"+{config.target_pct:.1%} 익절 ({config.ticker})"
        )
        return DipBuyResult(
            config=config,
            summary=_summarize(name, values, capital),
            liquidation=liquidation,
            cycles=cycles,
            events=events,
            equity_curve=[
                EquityPoint(date=ts.date(), equity=float(v))
                for ts, v in values.items()
            ],
            benchmark_after_tax=bh_after_tax,
        )
