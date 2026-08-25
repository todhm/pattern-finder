"""라오어 '무한매수법' 시뮬레이터 (v2.1 / v2.2).

일봉 OHLCV 전 구간 + (있으면) 15m 인트라데이 봉으로 체결을 판정한다:

- **LOC 매수/매도**는 정의상 종가 체결이므로 일봉 종가로 정확하다.
- **지정가 익절 매도**는 장중 터치 즉시 체결되므로 인트라데이가
  중요하다. 15m 봉이 있는 날은 봉 순서대로 스캔해 첫 도달 봉에서
  체결(갭 오픈이면 시가), 없는 날은 일봉 시가/고가 근사
  (시가 ≥ 목표 → 시가 체결, 고가 ≥ 목표 → 목표가 체결).
- 장중 전량 매도로 사이클이 끝난 날은 그날의 LOC 매수를 취소한다
  (사이클 종료 원칙).

`execute`는 정렬된 일봉 DataFrame(Open/High/Low/Close)과 선택적
15m DataFrame을 받는 순수 어댑터. 수수료·양도세는 다른 전략과 동일
(연 단위 정산 + 최종 청산).
"""

from __future__ import annotations

import pandas as pd

from strategy.adapters.band_rebalance_strategy import _summarize
from strategy.domain.models import (
    EquityPoint,
    InfiniteBuyingConfig,
    InfiniteBuyingCycle,
    InfiniteBuyingEvent,
    InfiniteBuyingPoint,
    InfiniteBuyingResult,
    LiquidationSummary,
    PortfolioSummary,
)


class InfiniteBuyingStrategy:
    """40분할 LOC 매집 + 지정가 익절 사이클 시뮬레이터."""

    def execute(
        self,
        daily: pd.DataFrame,
        config: InfiniteBuyingConfig,
        intraday: pd.DataFrame | None = None,
    ) -> InfiniteBuyingResult:
        df = daily.dropna(subset=["Open", "High", "Close"])
        if len(df) < 2:
            raise ValueError(
                f"Need at least 2 daily bars to run the backtest (got {len(df)})"
            )

        # 15m 봉을 날짜별로 묶는다 (정규장 필터는 composition root 몫).
        intraday_by_date: dict = {}
        if intraday is not None and len(intraday):
            ii = intraday.dropna(subset=["Open", "High"])
            keys = (
                ii.index.tz_localize(None) if ii.index.tz is not None else ii.index
            ).normalize()
            intraday_by_date = {
                d: g for d, g in ii.groupby(keys)
            }

        fee = config.fee_schedule
        sell_rate = fee.sell_commission_pct + fee.sec_fee_pct
        target = config.target_profit_pct
        is_v22 = config.version == "v2.2"

        # --- 계좌 상태 ---
        cash = config.initial_capital
        shares = 0.0
        notional_cost = 0.0  # 평단 계산용 (수수료 제외 매수 노셔널 합)
        tax_cost = 0.0  # 세무 취득원가 (수수료 포함)
        realized_ytd = 0.0
        realized_total = 0.0
        total_fees = 0.0
        total_tax = 0.0

        # --- 사이클 상태 ---
        cycle_no = 0
        cycle_capital = 0.0
        tranche = 0.0
        spent = 0.0
        cycle_start = None
        cycle_days = 0
        cycle_depleted = False
        in_cycle = False

        events: list[InfiniteBuyingEvent] = []
        cycles: list[InfiniteBuyingCycle] = []
        curve: list[InfiniteBuyingPoint] = []
        intraday_days = 0
        prev_year = df.index[0].year

        buys_in_cycle = 0

        def avg_price() -> float:
            return notional_cost / shares if shares > 0 else 0.0

        def record(ts, kind, price, notional_, qty=0.0, intraday_=False):
            ap = avg_price()
            events.append(
                InfiniteBuyingEvent(
                    ts=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    kind=kind,
                    price=price,
                    notional=notional_,
                    qty=qty,
                    shares_after=shares,
                    avg_price_after=ap,
                    cash_after=cash,
                    cycle_no=cycle_no,
                    tranche_no=buys_in_cycle if "buy" in kind else 0,
                    spent_pct=spent / cycle_capital if cycle_capital > 0 else 0.0,
                    target_price=ap * (1.0 + target) if shares > 0 else 0.0,
                    intraday=intraday_,
                )
            )

        def buy(ts, kind, amount, price):
            """현금 ``amount``(수수료 포함) 지출 매수."""
            nonlocal cash, shares, notional_cost, tax_cost, total_fees
            nonlocal spent, buys_in_cycle
            amount = min(amount, cash)
            if amount <= 0 or price <= 0:
                return
            notional = amount / (1.0 + fee.buy_commission_pct)
            shares_add = notional / price
            shares_ = shares + shares_add
            self_fees = amount - notional
            cash -= amount
            notional_cost += notional
            tax_cost += amount
            total_fees += self_fees
            spent += amount
            buys_in_cycle += 1
            # shares는 마지막에 갱신 (record가 최신 상태를 찍도록).
            _set_shares(shares_)
            record(ts, kind, price, notional, qty=shares_add)

        def _set_shares(v):
            nonlocal shares
            shares = v

        def sell(ts, kind, qty, price, intraday_=False):
            """``qty``주를 ``price``에 매도. 실현손익 기록."""
            nonlocal cash, shares, notional_cost, tax_cost
            nonlocal realized_ytd, realized_total, total_fees
            qty = min(qty, shares)
            if qty <= 0 or price <= 0:
                return
            notional = qty * price
            f = notional * sell_rate
            portion = qty / shares
            cost = tax_cost * portion
            gain = (notional - f) - cost
            realized_ytd += gain
            realized_total += gain
            notional_cost *= 1.0 - portion
            tax_cost -= cost
            cash += notional - f
            total_fees += f
            _set_shares(shares - qty)
            record(ts, kind, price, -notional, qty=qty, intraday_=intraday_)

        def end_cycle(ts, outcome):
            nonlocal in_cycle, cycle_depleted
            cycles.append(
                InfiniteBuyingCycle(
                    cycle_no=cycle_no,
                    start=cycle_start,
                    end=ts.date() if hasattr(ts, "date") else ts,
                    trading_days=cycle_days,
                    invested_max=spent,
                    pnl=cash - cycle_capital,
                    pnl_pct=(cash - cycle_capital) / cycle_capital
                    if cycle_capital > 0
                    else 0.0,
                    outcome=outcome,
                    depleted=cycle_depleted,
                )
            )
            in_cycle = False
            cycle_depleted = False

        for ts in df.index:
            o = float(df.loc[ts, "Open"])
            h = float(df.loc[ts, "High"])
            c = float(df.loc[ts, "Close"])

            # --- 연초 양도세 정산 ---
            if ts.year != prev_year:
                tax = max(0.0, realized_ytd - config.tax_deduction)
                tax *= config.capital_gains_tax_pct
                realized_ytd = 0.0
                if tax > 0:
                    if cash < tax and shares > 0:
                        sell(ts, "tax", (tax - cash) / (1.0 - sell_rate) / c, c)
                    cash -= tax
                    total_tax += tax
                    record(ts, "tax", c, -tax)
                prev_year = ts.year

            # --- 새 사이클 시작 (첫날 1T 종가 매수) ---
            if not in_cycle:
                cycle_no += 1
                cycle_capital = cash
                tranche = cycle_capital / config.divisions
                spent = 0.0
                buys_in_cycle = 0
                cycle_start = ts.date()
                cycle_days = 0
                in_cycle = True
                buy(ts, "start_buy", tranche, c)
                curve.append(self._point(ts, shares, c, cash, avg_price(), spent, cycle_capital))
                continue

            cycle_days += 1
            ap = avg_price()
            sold_out = False

            if shares > 0 and ap > 0:
                limit_price = ap * (1.0 + target)
                limit_qty = shares * 0.75 if is_v22 else shares
                quarter_qty = shares * 0.25 if is_v22 else 0.0

                # 1) 장중 지정가 익절 — 15m 있으면 봉 스캔, 없으면 근사.
                bars = intraday_by_date.get(ts.normalize())
                if bars is not None:
                    intraday_days += 1
                    for bts, bar in bars.iterrows():
                        if float(bar["High"]) >= limit_price:
                            fill = max(float(bar["Open"]), limit_price)
                            sell(bts, "limit_sell", limit_qty, fill, True)
                            break
                elif o >= limit_price:
                    sell(ts, "limit_sell", limit_qty, o)
                elif h >= limit_price:
                    sell(ts, "limit_sell", limit_qty, limit_price)

                # 2) 종가: 쿼터 LOC 매도 (v2.2).
                if is_v22 and shares > 0 and c >= ap * (1.0 + target / 2.0):
                    sell(ts, "quarter_sell", min(quarter_qty, shares), c)

                sold_out = shares <= 1e-9

            if sold_out:
                # 전량 청산 → 그날 LOC 매수 취소, 사이클 종료.
                end_cycle(ts, "profit")
            elif in_cycle and shares >= 0:
                remaining = cycle_capital - spent
                if remaining > 1e-6:
                    # 3) 종가 LOC 매수: 큰수 T/2 + 평단 LOC T/2.
                    buy(ts, "big_buy", min(tranche / 2.0, remaining), c)
                    remaining = cycle_capital - spent
                    if remaining > 1e-6 and c <= avg_price():
                        buy(ts, "avg_buy", min(tranche / 2.0, remaining), c)
                else:
                    cycle_depleted = True
                    if config.depletion_mode == "stop_loss":
                        sell(ts, "stop_loss", shares, c)
                        end_cycle(ts, "stop_loss")

            curve.append(
                self._point(ts, shares, c, cash, avg_price(), spent, cycle_capital)
            )

        # 진행 중 사이클 기록.
        if in_cycle:
            cycles.append(
                InfiniteBuyingCycle(
                    cycle_no=cycle_no,
                    start=cycle_start,
                    end=None,
                    trading_days=cycle_days,
                    invested_max=spent,
                    pnl=0.0,
                    pnl_pct=0.0,
                    outcome="open",
                    depleted=cycle_depleted,
                )
            )

        # --- 최종 청산 ---
        c_last = float(df["Close"].iloc[-1])
        pre_tax = curve[-1].total
        sale_notional = shares * c_last
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

        values = pd.Series([pt.total for pt in curve], index=df.index)
        name = (
            f"무한매수법 {config.version} — {config.ticker} "
            f"{config.divisions}분할, +{target:.0%} 익절"
            + (", 소진 손절" if config.depletion_mode == "stop_loss" else ", 소진 홀드")
        )

        # --- 벤치마크: 전액 B&H ---
        capital = config.initial_capital
        close = df["Close"]
        bh = close / float(close.iloc[0]) * (
            capital / (1.0 + fee.buy_commission_pct)
        )
        bench_name = f"{config.ticker} 100%"
        f_ = float(bh.iloc[-1]) * sell_rate
        gain_ = max(0.0, float(bh.iloc[-1]) - f_ - capital - config.tax_deduction)
        benchmarks = [_summarize(bench_name, bh, capital)]
        benchmark_curves = {
            bench_name: [
                EquityPoint(date=ts.date(), equity=float(v))
                for ts, v in bh.items()
            ]
        }
        benchmark_after_tax = {
            bench_name: float(bh.iloc[-1])
            - f_
            - gain_ * config.capital_gains_tax_pct
        }

        return InfiniteBuyingResult(
            config=config,
            curve=curve,
            events=events,
            cycles=cycles,
            summary=_summarize(name, values, capital),
            liquidation=liquidation,
            benchmarks=benchmarks,
            benchmark_curves=benchmark_curves,
            benchmark_after_tax=benchmark_after_tax,
            intraday_days=intraday_days,
        )

    @staticmethod
    def _point(ts, shares, price, cash, ap, spent, cycle_capital):
        return InfiniteBuyingPoint(
            date=ts.date(),
            total=shares * price + cash,
            stock_value=shares * price,
            cash=cash,
            avg_price=ap,
            tranches_spent_pct=spent / cycle_capital if cycle_capital > 0 else 0.0,
        )
