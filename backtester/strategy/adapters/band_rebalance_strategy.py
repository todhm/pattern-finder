"""VOO + TQQQ(QLD) 50:50 밴드 리밸런싱 전략.

docs/newstrategy의 전략을 구현한다:

- 초기 자본을 공격(TQQQ/QLD) : 방어(VOO) = 50:50으로 분할 매수,
  공격 자산 매수가를 **기준가**로 기록.
- 공격 자산이 기준가 대비 -15% → **줍줍 모드**: 방어 자산 평가액의
  15%를 팔아 공격 자산 매수, 기준가 갱신.
- 공격 자산이 기준가 대비 +15% → **수익 실현 모드**: 총 평가액을
  50:50으로 완전 리밸런싱, 기준가 갱신.

`execute`는 이미 정렬된 두 개의 종가 Series만 받는 순수 함수형
어댑터다 — 데이터 fetch / 합성 시계열 구성은 composition root
(Streamlit 페이지)의 몫. 트리거 판정과 체결 모두 당일 종가 기준
(원문 전략이 알람 기반 즉시 체결이므로 look-ahead 없음).
"""

from __future__ import annotations

import math

import pandas as pd

from strategy.domain.models import (
    BandRebalanceConfig,
    BandRebalanceResult,
    EquityPoint,
    PortfolioPoint,
    PortfolioSummary,
    RebalanceEvent,
)

# 합성 레버리지 시계열의 기본 연간 보수율 (TQQQ 0.84%, QLD 0.95%
# 수준 — UI에서 조정 가능).
DEFAULT_ANNUAL_EXPENSE = 0.0095
TRADING_DAYS_PER_YEAR = 252

# 실제 ETF 겹침 구간에서 실측한 총 드래그 (보수 + 스왑 스프레드,
# 자금조달비용 (L−1)×T-bill 은 별도 차감 후 남는 잔차):
#   TQQQ: r = 3·r_QQQ − 2·Tbill/252 − drag/252 로 회귀 시
#         2010-02~2026-07 (4,132일) 잔차 연 2.51%
#   QLD : 동일 방식 2006-06~2026-07 (5,049일) 잔차 연 1.65%
# 같은 모델 구조를 실존 2x 뮤추얼펀드 UOPIX(1997~)의 닷컴 구간에
# 적용하면 1997-12~2010-02 실제 0.327배 vs 모델 0.329배로 재현됨
# (일수익률 상관 0.974) — 상장 이전 백캐스트의 구조 검증 근거.
CALIBRATED_DRAG = {3.0: 0.0251, 2.0: 0.0165}


def build_synthetic_leveraged(
    close: pd.Series,
    leverage: float,
    annual_expense: float = DEFAULT_ANNUAL_EXPENSE,
    financing_rate: pd.Series | None = None,
) -> pd.Series:
    """기초지수 종가로 일간 N배 레버리지 ETF 가격을 합성한다.

    ``lev_ret[t] = N×ret[t] − (N−1)×financing[t]/252 − expense/252``

    - ``financing_rate``: 연환산 단기금리 시계열 (소수, 예: ^IRX/100).
      레버리지 ETF는 자기자본의 (N−1)배를 단기금리로 조달(스왑)하므로
      이 비용이 빠지면 고금리 시대(2000년 6%대)를 크게 과대평가한다.
      **실제 T-bill 데이터를 넘길 것** — None이면 미차감 (구버전 호환).
    - ``annual_expense``: 자금조달비용 차감 후 남는 총 드래그
      (보수 + 스왑 스프레드). 실측 보정값은 :data:`CALIBRATED_DRAG`
      참조 — 이 조합으로 2010~2026 실제 TQQQ 328배 대비 합성 343배
      (16.4년 누적 오차 +4.5%)까지 재현된다.

    시작 가격은 기초지수 첫 종가로 정규화 (절대 스케일은 비중
    전략에서 무의미).
    """
    ret = close.pct_change().fillna(0.0)
    lev_ret = leverage * ret - annual_expense / TRADING_DAYS_PER_YEAR
    if financing_rate is not None:
        fin = financing_rate.reindex(ret.index, method="ffill").fillna(0.0)
        lev_ret -= (leverage - 1.0) * fin / TRADING_DAYS_PER_YEAR
    # 일간 -33.4% 미만의 극단 하락에서 3배 ETF는 0 밑으로 못 내려감
    # (실제로는 서킷브레이커/장중 리밸런싱으로 -100% 캡).
    lev_ret = lev_ret.clip(lower=-0.999)
    # 첫 봉은 수익률이 정의되지 않으므로 보수 차감 없이 시작가 고정.
    lev_ret.iloc[0] = 0.0
    return float(close.iloc[0]) * (1.0 + lev_ret).cumprod()


def splice_series(early: pd.Series, late: pd.Series | None) -> pd.Series:
    """실데이터(``late``)가 시작되는 시점부터는 실데이터의 수익률을,
    그 이전은 ``early``(합성)의 수익률을 이어붙인 단일 시계열.

    레벨은 ``early`` 시작가 기준으로 연속 — "실데이터가 존재하는
    모든 구간은 실데이터" 원칙의 구현. 예: 1999~2010 합성 TQQQ +
    2010~ 실제 TQQQ.
    """
    if late is None or late.empty:
        return early
    anchor = late.index[0]
    pre = early[early.index < anchor]
    if pre.empty:
        return late
    bridge_ret = early.pct_change().get(anchor)
    bridge = pd.Series(
        [0.0 if bridge_ret is None or pd.isna(bridge_ret) else float(bridge_ret)],
        index=[anchor],
    )
    r = pd.concat([pre.pct_change(), bridge, late.pct_change().iloc[1:]])
    r = r[~r.index.duplicated(keep="first")].sort_index().fillna(0.0)
    return float(pre.iloc[0]) * (1.0 + r).cumprod()


def _max_drawdown_pct(values: pd.Series) -> float:
    """Peak 대비 최대 하락률 (음수, e.g. -0.55 = -55%)."""
    if len(values) == 0:
        return 0.0
    running_max = values.cummax()
    drawdown = values / running_max - 1.0
    return float(drawdown.min())


def _cagr_pct(initial: float, final: float, n_days: int) -> float:
    """달력일 기준 연평균 복리 수익률."""
    if initial <= 0 or final <= 0 or n_days <= 0:
        return 0.0
    years = n_days / 365.25
    if years <= 0:
        return 0.0
    return math.exp(math.log(final / initial) / years) - 1.0


def _summarize(name: str, values: pd.Series, initial: float) -> PortfolioSummary:
    final = float(values.iloc[-1])
    n_days = (values.index[-1] - values.index[0]).days
    return PortfolioSummary(
        name=name,
        final_value=final,
        total_return_pct=final / initial - 1.0,
        cagr_pct=_cagr_pct(initial, final, n_days),
        max_drawdown_pct=_max_drawdown_pct(values),
    )


class BandRebalanceStrategy:
    """두 자산 밴드 리밸런싱 시뮬레이터."""

    def execute(
        self,
        aggressive_close: pd.Series,
        defensive_close: pd.Series,
        config: BandRebalanceConfig,
        regime_close: pd.Series | None = None,
        risk_on_series: pd.Series | None = None,
    ) -> BandRebalanceResult:
        """정렬된 두 종가 Series로 전략 + 벤치마크를 시뮬레이션한다.

        두 Series는 같은 DatetimeIndex를 공유해야 하며(페이지에서
        inner-join 정렬), 첫 봉 종가에 초기 매수가 이뤄진다.

        ``regime_close``를 넘기면 **하락장 방어 레짐 필터**가 켜진다:
        레짐 지수(예: QQQ)가 ``regime_sma_days`` SMA 아래로
        ``regime_buffer_pct`` 이상 이탈하면 risk-off로 전환해
        ``risk_off_mode``의 방어 행동을 취하고, SMA 위로
        ``regime_buffer_pct`` 이상 복귀하면 목표 비중으로 재진입한다.
        SMA 수렴을 위해 시작일 이전 워밍업 구간을 포함해서 넘길 것.

        ``risk_on_series``는 **외부에서 계산한 risk-on 불리언**을
        직접 주입하는 경로 (거시 지표 합성 스코어 등 —
        :mod:`strategy.adapters.macro_regime` 참조). 히스테리시스/
        확인 일수는 upstream에서 이미 적용된 것으로 간주하고 상태
        전이만 따른다. ``regime_close``보다 우선한다.
        """
        joined = pd.DataFrame(
            {"agg": aggressive_close, "def": defensive_close}
        ).dropna()
        if len(joined) < 2:
            raise ValueError(
                "Need at least 2 overlapping bars to run the backtest "
                f"(got {len(joined)})"
            )

        agg = joined["agg"]
        dfn = joined["def"]
        capital = config.initial_capital
        w = config.aggressive_weight
        band = config.band_pct
        buf = config.regime_buffer_pct

        # 레짐 지수를 백테스트 달력에 정렬 (휴장/결측은 직전값 유지).
        regime: pd.DataFrame | None = None
        if regime_close is not None:
            sma = regime_close.rolling(config.regime_sma_days).mean()
            regime = pd.DataFrame(
                {"close": regime_close, "sma": sma}
            ).reindex(joined.index, method="ffill")

        # 외부 주입 risk-on 시계열 정렬 (있으면 SMA 경로보다 우선).
        flags: pd.Series | None = None
        if risk_on_series is not None:
            flags = risk_on_series.reindex(joined.index, method="ffill")

        # --- Day 0: 초기 50:50 매수, 기준가 기록 ---
        agg_shares = capital * w / float(agg.iloc[0])
        def_shares = capital * (1.0 - w) / float(dfn.iloc[0])
        ref_price = float(agg.iloc[0])
        cash = 0.0
        risk_on = True
        confirm_streak = 0  # risk-off 중 SMA 위 연속 유지 일수

        curve: list[PortfolioPoint] = []
        events: list[RebalanceEvent] = []

        for ts in joined.index:
            p = float(agg.loc[ts])
            d = float(dfn.loc[ts])
            transitioned = False

            # --- 1) 레짐 상태 전이 (당일 종가 체결) ---
            # 목표 상태 결정: 외부 주입 시계열(거시 합성 스코어 등,
            # 히스테리시스는 upstream 책임) 우선, 없으면 SMA 경로.
            desired: bool | None = None
            if ts != joined.index[0]:
                if flags is not None:
                    fv = flags.loc[ts]
                    if not pd.isna(fv):
                        desired = bool(fv)
                elif regime is not None:
                    r_close = float(regime.loc[ts, "close"])
                    r_sma = float(regime.loc[ts, "sma"])
                    if not math.isnan(r_sma):
                        if risk_on and r_close < r_sma * (1.0 - buf):
                            desired = False
                        elif not risk_on:
                            if r_close > r_sma * (1.0 + buf):
                                confirm_streak += 1
                            else:
                                # 확인 기간 중 조건 붕괴 — 처음부터.
                                confirm_streak = 0
                            if confirm_streak > config.regime_confirm_days:
                                desired = True

            if desired is False and risk_on:
                risk_on = False
                transitioned = True
                confirm_streak = 0
                sold_agg = 0.0
                if config.risk_off_mode == "derisk_defensive":
                    # 공격 자산 전량 → 방어 자산 대피.
                    sold_agg = agg_shares * p
                    def_shares += sold_agg / d
                    agg_shares = 0.0
                elif config.risk_off_mode == "derisk_cash":
                    # 전 자산 현금 대피 (무수익).
                    sold_agg = agg_shares * p
                    cash += sold_agg + def_shares * d
                    agg_shares = 0.0
                    def_shares = 0.0
                # "pause_dip": 보유 유지, 줍줍만 중단.
                events.append(
                    self._event(
                        ts, "risk_off", p, d, ref_price,
                        -sold_agg, agg_shares, def_shares,
                        cash=cash,
                    )
                )
            elif desired is True and not risk_on:
                risk_on = True
                transitioned = True
                confirm_streak = 0
                if config.risk_off_mode != "pause_dip":
                    # 재진입: 총액을 목표 비중으로 재배분.
                    total = agg_shares * p + def_shares * d + cash
                    bought = total * w - agg_shares * p
                    agg_shares = total * w / p
                    def_shares = total * (1.0 - w) / d
                    cash = 0.0
                    ref_before, ref_price = ref_price, p
                    events.append(
                        self._event(
                            ts, "risk_on", p, d, ref_before,
                            bought, agg_shares, def_shares,
                        )
                    )

            # --- 2) 밴드 트리거 (레짐 전이 봉은 건너뜀) ---
            band_allowed = ts != joined.index[0] and not transitioned
            # risk-off 중에는 줍줍 금지. pause_dip 모드는 보유를
            # 유지하므로 수익 실현(위험 축소 방향)만 계속 허용.
            dip_allowed = band_allowed and risk_on
            take_allowed = band_allowed and (
                risk_on or config.risk_off_mode == "pause_dip"
            )

            if dip_allowed and p <= ref_price * (1.0 - band):
                # 줍줍 모드: 방어 평가액의 dip_sell_defensive_pct
                # 만큼 팔아 싸진 공격 자산을 매수.
                sell_amount = def_shares * d * config.dip_sell_defensive_pct
                def_shares -= sell_amount / d
                agg_shares += sell_amount / p
                ref_before, ref_price = ref_price, p
                events.append(
                    self._event(
                        ts, "dip_buy", p, d, ref_before,
                        sell_amount, agg_shares, def_shares, cash=cash,
                    )
                )
            elif take_allowed and p >= ref_price * (1.0 + band):
                # 수익 실현 모드: 총액을 목표 비율로 완전 리밸런싱.
                total = agg_shares * p + def_shares * d
                traded = agg_shares * p - total * w  # 공격→방어 이동액
                agg_shares = total * w / p
                def_shares = total * (1.0 - w) / d
                ref_before, ref_price = ref_price, p
                events.append(
                    self._event(
                        ts, "profit_take", p, d, ref_before,
                        -traded, agg_shares, def_shares, cash=cash,
                    )
                )

            curve.append(
                PortfolioPoint(
                    date=ts.date(),
                    total=agg_shares * p + def_shares * d + cash,
                    aggressive_value=agg_shares * p,
                    defensive_value=def_shares * d,
                    reference_price=ref_price,
                    cash=cash,
                    risk_on=risk_on,
                )
            )

        strategy_values = pd.Series(
            [pt.total for pt in curve], index=joined.index
        )

        # --- 벤치마크: 첫 봉 종가 전액 매수 후 방치 ---
        bench_defs = {
            f"{config.defensive_ticker} 100%": dfn / float(dfn.iloc[0]),
            f"{config.aggressive_ticker} 100%": agg / float(agg.iloc[0]),
            "50:50 방치 (리밸런싱 없음)": (
                w * agg / float(agg.iloc[0])
                + (1.0 - w) * dfn / float(dfn.iloc[0])
            ),
        }
        benchmarks: list[PortfolioSummary] = []
        benchmark_curves: dict[str, list[EquityPoint]] = {}
        for name, norm in bench_defs.items():
            values = norm * capital
            benchmarks.append(_summarize(name, values, capital))
            benchmark_curves[name] = [
                EquityPoint(date=ts.date(), equity=float(v))
                for ts, v in values.items()
            ]

        strategy_name = (
            f"{config.defensive_ticker} {1 - w:.0%} + "
            f"{config.aggressive_ticker} {w:.0%} "
            f"({band:.0%} 리밸런싱)"
        )
        if flags is not None:
            strategy_name += " + 거시 레짐 방어"
        elif regime is not None:
            strategy_name += f" + {config.regime_sma_days}SMA 방어"
        return BandRebalanceResult(
            config=config,
            curve=curve,
            events=events,
            summary=_summarize(strategy_name, strategy_values, capital),
            benchmarks=benchmarks,
            benchmark_curves=benchmark_curves,
        )

    @staticmethod
    def _event(
        ts: pd.Timestamp,
        kind: str,
        agg_price: float,
        def_price: float,
        ref_before: float,
        traded_amount: float,
        agg_shares: float,
        def_shares: float,
        cash: float = 0.0,
    ) -> RebalanceEvent:
        agg_value = agg_shares * agg_price
        def_value = def_shares * def_price
        total = agg_value + def_value + cash
        return RebalanceEvent(
            date=ts.date(),
            kind=kind,
            aggressive_price=agg_price,
            reference_price_before=ref_before,
            traded_amount=traded_amount,
            aggressive_value_after=agg_value,
            defensive_value_after=def_value,
            total_value_after=total,
            aggressive_weight_after=agg_value / total if total > 0 else 0.0,
        )
