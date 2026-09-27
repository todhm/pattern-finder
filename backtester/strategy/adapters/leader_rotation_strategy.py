"""리더 종목 로테이션 — '폭발 성장주 갈아타기' 방법론 백테스트.

다바스 박스, 잰저의 차트 패턴, 미너비니 SEPA, 쿨라매기 브레이크아웃의
공통 뼈대(강한 놈만 산다 / 약해지면 갈아탄다 / 약세장엔 쉰다)를 일봉
데이터로 검증 가능한 형태로 근사한 어댑터. 규칙 상세는
:class:`LeaderRotationConfig` 참조.

`execute`는 조정종가 wide DataFrame(날짜×심볼)과 20일 평균 거래대금
wide DataFrame, 선택적 시장 지수 Series만 받는다 (데이터 소스 무관).
"""

from __future__ import annotations

import pandas as pd

from strategy.adapters.band_rebalance_strategy import _summarize
from strategy.domain.models import (
    EquityPoint,
    LeaderRebalance,
    LeaderRotationConfig,
    LeaderRotationResult,
    LiquidationSummary,
)


class LeaderRotationStrategy:
    """모멘텀 상위 종목 로테이션 시뮬레이터."""

    def execute(
        self,
        close: pd.DataFrame,
        dollar_vol: pd.DataFrame,
        config: LeaderRotationConfig,
        market: pd.Series | None = None,
    ) -> LeaderRotationResult:
        px = close.sort_index()
        mom = px.shift(config.skip_days) / px.shift(config.lookback_days) - 1.0
        if config.rank_mode == "vol_adj":
            vol = px.pct_change().rolling(config.lookback_days).std()
            rank_score = mom / (vol * (252.0**0.5)).replace(0.0, float("nan"))
        else:
            rank_score = mom
        stop_ma = (
            px.rolling(config.stop_ma_days).mean()
            if config.stop_ma_days > 0
            else None
        )
        # 52주 미만 상장 종목은 상장 후 고점을 기준으로 삼는다.
        hi52 = px.rolling(252, min_periods=1).max()
        ma = (
            px.rolling(config.trend_ma_days).mean()
            if config.trend_ma_days > 0
            else None
        )
        dv = dollar_vol.reindex(px.index).ffill()
        last_valid = {c: px[c].last_valid_index() for c in px.columns}

        market_ok = None
        if config.market_filter and market is not None:
            m = market.dropna()
            m_ma = m.rolling(200).mean()
            market_ok = (m > m_ma).reindex(px.index).ffill().fillna(True)

        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)
        dates = px.index[(px.index >= start) & (px.index <= end)]
        if len(dates) < 2:
            raise ValueError("백테스트 구간에 거래일이 부족함")

        fee = config.fee_schedule
        slip = config.slippage_bp / 10_000.0
        buy_rate = fee.buy_commission_pct + slip
        sell_rate = fee.sell_commission_pct + fee.sec_fee_pct + slip

        cash = config.initial_capital
        pos: dict[str, list[float]] = {}  # sym -> [shares, avg_cost, last_px]
        realized_ytd = 0.0
        realized_total = 0.0
        total_fees = 0.0
        total_tax = 0.0
        total_turnover = 0.0

        equity_curve: list[float] = []
        rebalances: list[LeaderRebalance] = []
        holdings_days = 0
        prev_year = dates[0].year
        next_rebalance = 0

        def sell(sym: str, price: float) -> float:
            nonlocal cash, realized_ytd, realized_total, total_fees
            shares, avg, _ = pos.pop(sym)
            notional = shares * price
            cost = notional * sell_rate
            realized = (notional - cost) - shares * avg
            realized_ytd += realized
            realized_total += realized
            cash += notional - cost
            total_fees += cost
            return notional

        def buy(sym: str, amount: float, price: float) -> float:
            nonlocal cash, total_fees
            amount = min(amount, cash)
            if amount <= 0 or price <= 0:
                return 0.0
            net = amount / (1.0 + buy_rate)
            shares = net / price
            if sym in pos:
                s0, a0, _ = pos[sym]
                pos[sym] = [s0 + shares, (s0 * a0 + amount) / (s0 + shares), price]
            else:
                pos[sym] = [shares, amount / shares, price]
            cash -= amount
            total_fees += amount - net
            return amount

        for di, ts in enumerate(dates):
            row = px.loc[ts]
            # 보유 종목 가격 갱신 + 데이터 끊긴 종목 강제 청산.
            for sym in list(pos):
                p = row.get(sym)
                if pd.notna(p) and p > 0:
                    pos[sym][2] = float(p)
                elif last_valid[sym] is None or ts > last_valid[sym]:
                    sell(sym, pos[sym][2])  # 상폐/데이터 종료 — 마지막가 청산
                    continue
                # 빠른 손절: N일선 종가 이탈 즉시 매도 (다음 랭킹까지 현금).
                if stop_ma is not None and sym in pos:
                    s = stop_ma.loc[ts, sym]
                    if pd.notna(s) and pos[sym][2] < float(s):
                        sell(sym, pos[sym][2])

            # 연초 양도세 정산.
            if ts.year != prev_year:
                tax = max(0.0, realized_ytd - config.tax_deduction)
                tax *= config.capital_gains_tax_pct
                realized_ytd = 0.0
                if tax > 0:
                    while cash < tax and pos:
                        sym = next(iter(pos))
                        sell(sym, pos[sym][2])
                    cash -= tax
                    total_tax += tax
                prev_year = ts.year

            # --- 로테이션 ---
            if di >= next_rebalance:
                next_rebalance = di + config.rebalance_days
                risk_on = True
                if market_ok is not None and not bool(market_ok.loc[ts]):
                    risk_on = False

                picks: list[str] = []
                mom_row = mom.loc[ts]
                score_row = rank_score.loc[ts]
                if risk_on:
                    elig = score_row.notna() & row.notna()
                    elig &= row >= config.min_price
                    elig &= dv.loc[ts].fillna(0.0) >= config.min_dollar_volume
                    if ma is not None:
                        elig &= row > ma.loc[ts]
                    elig &= row >= hi52.loc[ts] * (1.0 - config.near_high_pct)
                    ranked = score_row[elig].sort_values(ascending=False)
                    picks = list(ranked.index[: config.top_n])

                traded = 0.0
                for sym in list(pos):
                    if sym not in picks:
                        traded += sell(sym, pos[sym][2])
                equity = cash + sum(s * p for s, _, p in pos.values())
                if picks:
                    target = equity / config.top_n
                    for sym in picks:
                        p = float(row[sym])
                        held = pos[sym][0] * p if sym in pos else 0.0
                        if held > target * 1.02:
                            shares, avg, _ = pos[sym]
                            excess = held - target
                            qty = excess / p
                            pos[sym] = [shares - qty, avg, p]
                            cost = excess * sell_rate
                            gain = (excess - cost) - qty * avg
                            realized_ytd += gain
                            realized_total += gain
                            cash += excess - cost
                            total_fees += cost
                            traded += excess
                        elif held < target * 0.98:
                            traded += buy(sym, target - held, p)
                if equity > 0:
                    total_turnover += traded / equity
                rebalances.append(
                    LeaderRebalance(
                        date=ts.date(),
                        picks=picks,
                        momentum={
                            s: float(mom_row[s])
                            for s in picks
                            if pd.notna(mom_row.get(s))
                        },
                        regime="risk_on" if risk_on else "risk_off",
                        turnover=traded / equity if equity > 0 else 0.0,
                    )
                )

            holdings_days += len(pos)
            equity_curve.append(cash + sum(s * p for s, _, p in pos.values()))

        values = pd.Series(equity_curve, index=dates)
        pre_tax = float(values.iloc[-1])
        # 최종 청산: 잔여 포지션 매도 가정 후 미실현 이익 과세.
        final_fee = 0.0
        final_gain = 0.0
        for shares, avg, p in pos.values():
            notional = shares * p
            f = notional * sell_rate
            final_fee += f
            final_gain += (notional - f) - shares * avg
        taxable = max(0.0, realized_ytd + final_gain - config.tax_deduction)
        final_tax = taxable * config.capital_gains_tax_pct
        liquidation = LiquidationSummary(
            final_value_pre_tax=pre_tax,
            final_value_after_tax=pre_tax - final_fee - final_tax,
            final_tax=final_tax,
            total_interest=0.0,
            total_fees=total_fees + final_fee,
            total_tax=total_tax + final_tax,
            realized_gain_total=realized_total + final_gain,
        )

        yearly: dict[int, float] = {}
        for year, grp in values.groupby(values.index.year):
            yearly[int(year)] = float(grp.iloc[-1] / grp.iloc[0] - 1.0)

        name = (
            f"리더 로테이션 (모멘텀 {config.lookback_days}일 상위 "
            f"{config.top_n}종목, {config.rebalance_days}일마다)"
        )
        return LeaderRotationResult(
            config=config,
            summary=_summarize(name, values, config.initial_capital),
            liquidation=liquidation,
            equity_curve=[
                EquityPoint(date=t.date(), equity=float(v))
                for t, v in values.items()
            ],
            rebalances=rebalances,
            yearly_returns=yearly,
            avg_holdings=holdings_days / max(len(dates), 1),
            total_turnover=total_turnover,
        )


def scan_leaders(
    close: pd.DataFrame,
    dollar_vol: pd.DataFrame,
    config: LeaderRotationConfig,
    market: pd.Series | None = None,
    asof: pd.Timestamp | None = None,
) -> dict:
    """오늘(또는 asof) 기준 리더 랭킹 스냅샷 — 실행(시그널) 페이지용.

    `LeaderRotationStrategy.execute`의 리밸런스일 랭킹과 동일한 로직.
    반환: {"date", "regime", "rows": [{symbol, momentum, score, price,
    dollar_vol, pct_from_high} ...]} — rows는 점수 내림차순 top_n.
    """
    px = close.sort_index()
    # 벌크 다운로드 청크 간 시점 차이로 맨 끝 행은 일부 종목만 값이
    # 있을 수 있다 — 커버리지가 최대치의 절반 이상인 마지막 날을 기준.
    cov = px.notna().sum(axis=1)
    good = px.index[cov >= cov.max() * 0.5]
    ts = asof if asof is not None else good[-1]
    ts = good[good <= ts][-1]

    mom = px.shift(config.skip_days) / px.shift(config.lookback_days) - 1.0
    if config.rank_mode == "vol_adj":
        vol = px.pct_change().rolling(config.lookback_days).std()
        score = mom / (vol * (252.0**0.5)).replace(0.0, float("nan"))
    else:
        score = mom
    hi52 = px.rolling(252, min_periods=1).max()
    row = px.loc[ts]

    regime = "risk_on"
    if config.market_filter and market is not None:
        m = market.dropna()
        m_ma = m.rolling(200).mean()
        ok = (m > m_ma).reindex(px.index).ffill().fillna(True)
        if not bool(ok.loc[ts]):
            regime = "risk_off"

    elig = score.loc[ts].notna() & row.notna()
    elig &= row >= config.min_price
    dv_row = dollar_vol.reindex(px.index).ffill().loc[ts].fillna(0.0)
    elig &= dv_row >= config.min_dollar_volume
    if config.trend_ma_days > 0:
        elig &= row > px.rolling(config.trend_ma_days).mean().loc[ts]
    elig &= row >= hi52.loc[ts] * (1.0 - config.near_high_pct)

    ranked = score.loc[ts][elig].sort_values(ascending=False)
    rows = [
        {
            "symbol": sym,
            "momentum": float(mom.loc[ts, sym]),
            "score": float(ranked[sym]),
            "price": float(row[sym]),
            "dollar_vol": float(dv_row[sym]),
            "pct_from_high": float(row[sym] / hi52.loc[ts, sym] - 1.0),
        }
        for sym in ranked.index[: config.top_n]
    ]
    return {"date": ts.date(), "regime": regime, "rows": rows}
