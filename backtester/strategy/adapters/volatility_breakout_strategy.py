"""래리 윌리엄스 계열 변동성 돌파 데이트레이드 시뮬레이터.

1987년 월드컵 트레이딩 챔피언십 11,376%(1년 113배, 제3자 검증)의
방법론 계열: 시가 + k×전일 레인지 돌파 매수 → 당일 청산, 수익의
본체는 공격적 사이징(켈리). 이 시뮬레이터의 목적은 "엣지가
존재하는가"와 "레버리지를 올리면 연 10배가 나오는가 vs 파산하는가"
를 정량화하는 것이다.

규칙 상세는 :class:`VolBreakoutConfig` 참조. 갭 오픈(시가 > 트리거)
은 시가 체결. 레버리지의 자금조달 비용(펀딩비)은 빌린 부분에
트레이드당 1일치(기본 연 10%)를 부과한다.
"""

from __future__ import annotations

import pandas as pd

from strategy.adapters.band_rebalance_strategy import _summarize
from strategy.domain.models import (
    EquityPoint,
    LiquidationSummary,
    VolBreakoutConfig,
    VolBreakoutResult,
    VolBreakoutTrade,
)


class VolatilityBreakoutStrategy:
    """시가 + k×전일 레인지 돌파 → 당일 청산."""

    def execute(
        self, daily: pd.DataFrame, config: VolBreakoutConfig
    ) -> VolBreakoutResult:
        df = daily.dropna(subset=["Open", "High", "Low", "Close"])
        if len(df) < 3:
            raise ValueError(
                f"Need at least 3 daily bars to run the backtest (got {len(df)})"
            )

        fee = config.fee_schedule
        buy_cost = fee.buy_commission_pct + config.slippage_bp / 10_000.0
        sell_cost = (
            fee.sell_commission_pct
            + fee.sec_fee_pct
            + config.slippage_bp / 10_000.0
        )

        opens = df["Open"].to_numpy(float)
        highs = df["High"].to_numpy(float)
        lows = df["Low"].to_numpy(float)
        closes = df["Close"].to_numpy(float)
        index = df.index
        sma = (
            df["Close"].rolling(config.trend_filter_days).mean().to_numpy(float)
            if config.trend_filter_days > 0
            else None
        )

        equity = config.initial_capital
        floor = config.initial_capital * config.ruin_floor_pct
        # 빌린 부분((lev−1)×평가액)의 1일치 자금조달 비용 비율.
        financing_daily = (
            max(config.leverage - 1.0, 0.0)
            * config.financing_annual_rate / 365.0
        )
        realized_ytd = 0.0
        realized_total = 0.0
        total_fees = 0.0
        total_tax = 0.0
        total_financing = 0.0
        ruined = False
        trades: list[VolBreakoutTrade] = []
        curve: list[float] = []
        prev_year = index[0].year

        for i in range(len(df)):
            ts = index[i]
            # 연초 양도세 정산 (전액 현금이므로 그냥 차감).
            if ts.year != prev_year:
                tax = max(0.0, realized_ytd - config.tax_deduction)
                tax *= config.capital_gains_tax_pct
                realized_ytd = 0.0
                if tax > 0:
                    equity -= tax
                    total_tax += tax
                prev_year = ts.year

            if i >= 1 and not ruined:
                prev_range = highs[i - 1] - lows[i - 1]
                trigger = opens[i] + config.k * prev_range
                trend_ok = (
                    sma is None
                    or (not pd.isna(sma[i - 1]) and closes[i - 1] > sma[i - 1])
                )
                if trend_ok and prev_range > 0 and highs[i] >= trigger:
                    entry = max(opens[i], trigger)
                    if config.exit_mode == "next_open" and i + 1 < len(df):
                        exit_price = opens[i + 1]
                    else:
                        exit_price = closes[i]
                    notional = equity * config.leverage
                    gross = notional * (exit_price / entry - 1.0)
                    costs = notional * buy_cost + (
                        notional * (exit_price / entry)
                    ) * sell_cost
                    financing = equity * financing_daily
                    total_financing += financing
                    pnl = gross - costs - financing
                    equity += pnl
                    realized_ytd += pnl
                    realized_total += pnl
                    total_fees += costs
                    trades.append(
                        VolBreakoutTrade(
                            date=ts.date(),
                            entry=entry,
                            exit=exit_price,
                            ret_pct=pnl / notional,
                            pnl=pnl,
                            equity_after=max(equity, 0.0),
                        )
                    )
                    if equity <= floor:
                        ruined = True
                        equity = max(equity, 0.0)

            curve.append(max(equity, 0.0))

        values = pd.Series(curve, index=index)
        pre_tax = float(values.iloc[-1])
        # 전액 현금 종료 — 최종 청산 수수료 없음, 당해 실현분만 과세.
        taxable = max(0.0, realized_ytd - config.tax_deduction)
        final_tax = taxable * config.capital_gains_tax_pct
        liquidation = LiquidationSummary(
            final_value_pre_tax=pre_tax,
            final_value_after_tax=max(pre_tax - final_tax, 0.0),
            final_tax=final_tax,
            total_fees=total_fees,
            total_tax=total_tax + final_tax,
            realized_gain_total=realized_total,
        )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = (
            sum(t.ret_pct for t in wins) / len(wins) if wins else 0.0
        )
        avg_loss = (
            abs(sum(t.ret_pct for t in losses) / len(losses)) if losses else 0.0
        )
        kelly = 0.0
        if avg_loss > 0 and avg_win > 0:
            r = avg_win / avg_loss
            kelly = max(0.0, win_rate - (1.0 - win_rate) / r)

        yearly: dict[int, float] = {}
        by_year = values.groupby(values.index.year)
        for year, seg in by_year:
            if len(seg) > 1 and float(seg.iloc[0]) > 0:
                yearly[int(year)] = float(seg.iloc[-1] / seg.iloc[0] - 1.0)

        capital = config.initial_capital
        bh_final = closes[-1] / closes[0] * (
            capital / (1.0 + fee.buy_commission_pct)
        )
        bh_fee = bh_final * (fee.sell_commission_pct + fee.sec_fee_pct)
        bh_gain = max(0.0, bh_final - bh_fee - capital - config.tax_deduction)
        bench_after = bh_final - bh_fee - bh_gain * config.capital_gains_tax_pct

        name = (
            f"변동성 돌파 k={config.k:g} × {config.leverage:g}배 "
            f"({config.ticker}, {'당일 종가' if config.exit_mode == 'close' else '익일 시가'} 청산"
            + (f", {config.trend_filter_days}MA 필터" if config.trend_filter_days else "")
            + ")"
        )
        return VolBreakoutResult(
            config=config,
            summary=_summarize(name, values, capital),
            liquidation=liquidation,
            trades=trades,
            equity_curve=[
                EquityPoint(date=t.date(), equity=float(v))
                for t, v in values.items()
            ],
            yearly_returns=yearly,
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            kelly_fraction=kelly,
            total_financing=total_financing,
            ruined=ruined,
            benchmark_after_tax=bench_after,
        )
