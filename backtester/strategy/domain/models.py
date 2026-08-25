from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field


class StrategyConfig(BaseModel):
    ticker: str
    start_date: date
    end_date: date
    pattern_name: str
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.02
    max_holding_days: int = 60
    pattern_params: dict[str, Any] = Field(default_factory=dict)


class Trade(BaseModel):
    pattern_name: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    stop_loss: float
    shares: int
    pnl: float
    pnl_pct: float
    # Which exit condition closed this trade:
    #   "exhaustion_exit"   — injected ``exit_detector`` fired
    #   "trendline_break"   — higher-low trendline exit
    #   "smart_trail"       — Chandelier trail
    #   "resistance_break"  — false-breakout of entry-time swing resistance
    #   "breakeven_stop"    — broke-even stop after ≥ 1R unrealized gain
    #   "end_of_data"       — no rule fired; held to the last bar
    exit_reason: str = "end_of_data"
    # Intraday bar timestamps. Always populated by the strategy
    # layer (including the 1d path), so downstream consumers can
    # pick whichever precision they need. Daily pages that only
    # want the session date keep reading ``entry_date``/``exit_date``
    # unchanged; 15m pages read the ts fields for sub-day rendering.
    entry_ts: datetime | None = None
    exit_ts: datetime | None = None


class StrategyPerformance(BaseModel):
    initial_capital: float
    final_capital: float
    total_return_pct: float
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    trades: list[Trade]


class EquityPoint(BaseModel):
    date: date
    equity: float


class StrategyResult(BaseModel):
    config: StrategyConfig
    performance: StrategyPerformance
    equity_curve: list[EquityPoint]


class TossFeeSchedule(BaseModel):
    """Toss Securities (토스증권) US-stock fee schedule.

    Defaults reflect Toss's published US-equity fees:

    - **거래수수료 0.1%** on both buy and sell notional.
    - **SEC fee 0.00229%** on sell notional only — the US Section 31
      regulatory fee that all brokers pass through.

    FX spread / 환전 수수료 is *not* modeled here: yfinance returns
    USD prices, so the backtest stays in USD and currency conversion
    is treated as out-of-band.
    """

    buy_commission_pct: float = 0.001
    sell_commission_pct: float = 0.001
    sec_fee_pct: float = 0.0000229

    def buy_fee(self, price: float, shares: int) -> float:
        return price * shares * self.buy_commission_pct

    def sell_fee(self, price: float, shares: int) -> float:
        rate = self.sell_commission_pct + self.sec_fee_pct
        return price * shares * rate

    def round_trip(
        self, entry_price: float, exit_price: float, shares: int
    ) -> float:
        return self.buy_fee(entry_price, shares) + self.sell_fee(
            exit_price, shares
        )


class MultiStrategyConfig(BaseModel):
    """Config for a single-portfolio scan over a ticker universe.

    The portfolio holds one position at a time. On any day with one or
    more wedge-pop signals across the universe, the highest-volume
    signal wins. While a position is open, every other signal is
    ignored until exit.
    """

    universe: str
    start_date: date
    end_date: date
    pattern_name: str = "wedge_pop"
    initial_capital: float = 100_000.0
    risk_per_trade: float = 0.02
    max_holding_days: int = 60
    max_tickers: int | None = None
    fee_schedule: TossFeeSchedule = Field(default_factory=TossFeeSchedule)


class MultiTrade(Trade):
    """A trade taken in a multi-ticker scan.

    Tagged with its source ticker, the signal-day buy/sell pressure
    that won the daily auction, and the Toss commission paid (already
    deducted from ``pnl`` / ``pnl_pct``).

    The ranking metric is ``signal_buy_sell_ratio`` — buy volume over
    sell volume on the signal bar, estimated from OHLC via the
    standard accumulation/distribution split. ``signal_volume`` is
    kept around as informational context but is no longer the
    selection key.
    """

    ticker: str
    signal_volume: float
    signal_buy_volume: float
    signal_sell_volume: float
    signal_buy_sell_ratio: float
    commission: float
    gross_pnl: float


class LiquidationSummary(BaseModel):
    """마지막 날 전량 청산 가정의 세후 결과와 누적 비용.

    수수료·양도세를 모델링하는 모든 전략이 공유한다.
    ``total_interest``는 이자 수취가 없는 전략에선 0.
    """

    final_value_pre_tax: float
    final_value_after_tax: float
    final_tax: float
    total_interest: float = 0.0
    total_fees: float
    total_tax: float
    realized_gain_total: float


# 하위 호환 별칭 — TqqqP2p 전용이던 시절의 이름.
TqqqP2pLiquidation = LiquidationSummary


class BandRebalanceConfig(BaseModel):
    """Config for the VOO+TQQQ(QLD) 50:50 band-rebalancing strategy.

    Rules (docs/newstrategy 기준):

    - 초기 자본을 ``aggressive_weight`` : ``1 - aggressive_weight``로
      분할 매수하고, 공격 자산의 매수가를 **기준가**로 기록한다.
    - 공격 자산 종가가 기준가 대비 ``band_pct`` 이상 **하락**하면
      (줍줍 모드) 방어 자산 평가액의 ``dip_sell_defensive_pct``만큼
      팔아 공격 자산을 사고, 기준가를 현재가로 갱신한다.
    - 기준가 대비 ``band_pct`` 이상 **상승**하면 (수익 실현 모드)
      총 평가액을 목표 비율로 완전 리밸런싱하고 기준가를 갱신한다.

    체결은 트리거 당일 **종가** 기준, 수량은 소수점 허용(비중 전략
    이므로 정수 주식 반올림 오차가 결과를 왜곡하지 않도록).
    """

    aggressive_ticker: str = "TQQQ"
    defensive_ticker: str = "VOO"
    start_date: date
    end_date: date
    initial_capital: float = 100_000_000.0
    band_pct: float = 0.15
    aggressive_weight: float = 0.5
    dip_sell_defensive_pct: float = 0.15
    # --- 하락장 방어 레짐 필터 (execute에 regime_close를 넘길 때만 동작) ---
    # 원 전략의 치명 구간(1999-2002 닷컴 버블: MDD -97%)은 하락장
    # 내내 방어 자산을 팔아 떨어지는 칼날을 계속 받는 구조에서 온다.
    # 레짐 지수(예: QQQ)가 SMA 아래로 내려가면 방어 태세로 전환한다.
    #
    # ``risk_off_mode``:
    #   - "derisk_defensive" — 공격 자산 전량을 방어 자산으로 대피
    #   - "derisk_cash"      — 전 자산 현금(무수익) 대피
    #   - "pause_dip"        — 보유 유지, 줍줍(하락 매수)만 중단
    regime_sma_days: int = 200
    regime_buffer_pct: float = 0.0
    risk_off_mode: str = "derisk_defensive"
    # 재진입 확인 일수: 레짐 지수가 N일 **연속** SMA 위를 유지해야
    # risk-on 복귀. 2000-02 같은 긴 하락장의 베어랠리 휩쏘(짧은
    # SMA 상향 돌파 → 재진입 → 다음 하락 다리 직격)를 걸러낸다.
    # 이탈(risk-off)은 즉시 — 방어는 빠르게, 재진입은 신중하게.
    regime_confirm_days: int = 0
    # --- 수수료·양도소득세 (기본 미반영 = 기존 동작과 동일) ---
    # fee_schedule을 넘기면 모든 매매에 수수료가 붙고, 실현 차익엔
    # 연 단위 양도세(기본공제 차감)가 부과된다. 세금은 현금 → 방어
    # 자산 매도 → 공격 자산 매도 순으로 납부. 결과의 ``liquidation``
    # 에 최종 전량 청산 가정의 세후 가치가 담긴다.
    fee_schedule: TossFeeSchedule | None = None
    capital_gains_tax_pct: float = 0.0
    tax_deduction: float = 2_500_000.0


class RebalanceEvent(BaseModel):
    """One triggered rebalancing action.

    ``kind``:
      - ``"dip_buy"``      — 기준가 대비 -band 하락: 방어 자산 일부
        매도 → 공격 자산 매수 (줍줍 모드)
      - ``"profit_take"``  — 기준가 대비 +band 상승: 목표 비율로
        완전 리밸런싱 (수익 실현 모드)
      - ``"risk_off"``     — 레짐 지수가 SMA 아래로 이탈: 방어 태세
        전환 (risk_off_mode에 따라 대피/줍줍 중단)
      - ``"risk_on"``      — 레짐 복귀: 목표 비율로 재진입, 기준가
        리셋

    ``traded_amount``는 공격 자산으로 이동한 금액(+) / 공격 자산에서
    빠져나간 금액(-).
    """

    date: date
    kind: str
    aggressive_price: float
    reference_price_before: float
    traded_amount: float
    aggressive_value_after: float
    defensive_value_after: float
    total_value_after: float
    aggressive_weight_after: float


class PortfolioPoint(BaseModel):
    """Daily mark-to-market snapshot of the two-asset portfolio."""

    date: date
    total: float
    aggressive_value: float
    defensive_value: float
    reference_price: float
    cash: float = 0.0
    risk_on: bool = True


class PortfolioSummary(BaseModel):
    """Headline stats for one portfolio variant (strategy or benchmark)."""

    name: str
    final_value: float
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float


class BandRebalanceResult(BaseModel):
    config: BandRebalanceConfig
    curve: list[PortfolioPoint]
    events: list[RebalanceEvent]
    summary: PortfolioSummary
    # Buy-and-hold comparisons over the identical window: 방어 100%,
    # 공격 100%, 그리고 리밸런싱 없는 50:50 방치.
    benchmarks: list[PortfolioSummary]
    benchmark_curves: dict[str, list[EquityPoint]]
    # 수수료·세금 모델링 시의 청산 요약 (fee_schedule 미설정이면
    # 수수료·세금 0으로 계산된 값 — after_tax == pre_tax).
    liquidation: LiquidationSummary | None = None

    @property
    def dip_buy_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "dip_buy")

    @property
    def profit_take_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "profit_take")


class TqqqP2pConfig(BaseModel):
    """Config for the TQQQ + P2P 채권 50:50 현금흐름 리밸런싱 전략.

    - 초기 자본을 ``tqqq_weight`` : 나머지로 TQQQ / P2P 채권에 분할.
    - 채권은 연 ``bond_annual_rate`` 이자를 **매월** 지급하고
      ``bond_maturity_months``(기본 12개월 = 1년 만기) 뒤 원금을
      상환하는 사다리(ladder).
    - 매월 첫 거래일에 이자·만기 원금이 현금으로 들어오고, 이 현금을
      목표 비중(50:50)에 모자란 쪽부터 최대로 투자한다. 남으면 새
      채권을 매입한다 (채권 매입엔 수수료 없음).
    - ``allow_sell_rebalance``: True면 TQQQ가 목표 비중을 초과할 때
      초과분을 **매도**해서까지 50:50을 강제한다 (매도 시 수수료 +
      양도차익 실현 → 과세). False(기본)면 현금흐름으로만 리밸런싱.
    - 매도 차익엔 연 단위로 ``capital_gains_tax_pct`` 양도소득세를
      부과 (연 ``tax_deduction`` 기본공제 차감 후). 채권 이자에 대한
      소득세는 모델링하지 않는다 (P2P 법인 운영 가정 — 법인세는
      out-of-band).
    """

    ticker: str = "TQQQ"
    start_date: date
    end_date: date
    initial_capital: float = 100_000_000.0
    tqqq_weight: float = 0.5
    bond_annual_rate: float = 0.09
    bond_maturity_months: int = 12
    allow_sell_rebalance: bool = False
    fee_schedule: TossFeeSchedule = Field(default_factory=TossFeeSchedule)
    capital_gains_tax_pct: float = 0.22
    tax_deduction: float = 2_500_000.0


class TqqqP2pEvent(BaseModel):
    """월별 현금흐름 처리 내역 (매월 첫 거래일 종가 기준).

    ``tqqq_traded``: TQQQ 매수 노셔널(+) / 매도 노셔널(−).
    """

    date: date
    interest: float
    matured_principal: float
    tax_paid: float
    tqqq_traded: float
    bond_invested: float
    tqqq_value_after: float
    bond_value_after: float
    cash_after: float
    tqqq_weight_after: float


class TqqqP2pPoint(BaseModel):
    """일별 mark-to-market. 채권은 액면(원금) 기준 평가."""

    date: date
    total: float
    tqqq_value: float
    bond_value: float
    cash: float


class TqqqP2pResult(BaseModel):
    config: TqqqP2pConfig
    curve: list[TqqqP2pPoint]
    events: list[TqqqP2pEvent]
    summary: PortfolioSummary
    liquidation: LiquidationSummary
    benchmarks: list[PortfolioSummary]
    benchmark_curves: dict[str, list[EquityPoint]]
    # 벤치마크별 최종 청산 세후 가치 (이름 → 값).
    benchmark_after_tax: dict[str, float]


class ValueRebalanceConfig(BaseModel):
    """라오어 '밸류 리밸런싱(VR)' 전략 설정 (거치식 기준).

    - 초기 자본을 주식(``stock_ratio``) : Pool(현금)로 분할,
      V(밸류패스) 초기값 = 초기 주식 평가금.
    - ``cycle_days`` 거래일마다 ``V += Pool / G`` (거치식 — 적립/인출
      없음). Pool이 클수록 V가 빨리 오르는 자기조절 구조.
    - 평가금 E가 밴드 상단(V × (1+band))을 넘으면 초과분(E − V)을
      매도해 Pool에 적립, 하단(V × (1−band)) 아래면 부족분(V − E)을
      Pool 한도 내에서 매수.
    - ``check_daily``: True(기본)면 매일 밴드를 검사(책의 매수표·
      매도표 LOC 방식 근사), False면 사이클 시점에만 검사.
    - 수수료·양도세는 TQQQ P2P 전략과 동일하게 연 단위 정산 +
      최종 청산 과세.
    """

    ticker: str = "TQQQ"
    start_date: date
    end_date: date
    initial_capital: float = 100_000_000.0
    stock_ratio: float = 0.75
    gradient: float = 10.0
    cycle_days: int = 10
    band_pct: float = 0.15
    check_daily: bool = True
    # 실력공식(고급) 근사: V 증가분에 √(E/V) 보정 — 평가금이 밸류패스에
    # 못 미치는 하락장에선 V 상승을 늦춰 Pool 소진을 막고, 앞서가는
    # 상승장에선 가속한다. 책 원본 공식은 비공개(서적 전용)라 공개된
    # 설명("E와 V의 괴리를 루트로 보정")을 따른 근사임.
    advanced_formula: bool = False
    # 체결 목표: True(기본)면 밴드 **가장자리**까지만 복원 — 책의
    # 매수표·매도표 LOC 분할 체결과 등가 (종가가 한 단계 더 벗어나면
    # 그만큼만 추가 체결 = 가격대별 분할 매매). False면 V까지 한 번에
    # 당기는 공격적 체결 (초기 구현 호환).
    rebalance_to_edge: bool = True
    # Pool(현금)을 P2P 채권 등으로 굴릴 때의 연이율 (0 = 무수익 현금).
    # 일할 계산(연이율/252)으로 매 거래일 누적 — 즉시 인출 가능하다고
    # 가정하므로 1년 만기 락업의 유동성 제약은 반영하지 않는 낙관적
    # 상한임에 유의.
    pool_annual_rate: float = 0.0
    fee_schedule: TossFeeSchedule = Field(default_factory=TossFeeSchedule)
    capital_gains_tax_pct: float = 0.22
    tax_deduction: float = 2_500_000.0


class ValueRebalanceEvent(BaseModel):
    """VR 매매/사이클 이벤트.

    ``kind``: "sell"(상단 초과 매도) / "buy"(하단 이탈 매수) /
    "tax"(연초 양도세 정산). ``traded``는 매매 노셔널(매수 +/매도 −).
    """

    date: date
    kind: str
    traded: float
    value_path: float
    stock_value_after: float
    pool_after: float


class ValueRebalancePoint(BaseModel):
    """일별 스냅샷 — 평가금 E, Pool, 밸류패스 V."""

    date: date
    total: float
    stock_value: float
    pool: float
    value_path: float


class ValueRebalanceResult(BaseModel):
    config: ValueRebalanceConfig
    curve: list[ValueRebalancePoint]
    events: list[ValueRebalanceEvent]
    summary: PortfolioSummary
    liquidation: LiquidationSummary
    benchmarks: list[PortfolioSummary]
    benchmark_curves: dict[str, list[EquityPoint]]
    benchmark_after_tax: dict[str, float]

    @property
    def buy_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "buy")

    @property
    def sell_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "sell")


class InfiniteBuyingConfig(BaseModel):
    """라오어 '무한매수법' 설정 (TQQQ 등 3배 레버리지 ETF 전용).

    - 사이클 원금을 ``divisions``(기본 40)분할, 1일 매수금 T = 원금/40.
    - 매일 LOC 주문 2건: **큰수 LOC** T/2(종가 무조건 체결) +
      **평단 LOC** T/2(종가 ≤ 평단일 때만 체결). 사이클 첫날은 1T.
    - 매도(v2.1): 전량 평단×(1+``target_profit_pct``) 지정가 GTC.
      (v2.2): 보유량 25%는 평단×(1+target/2) **LOC 매도**, 75%는
      평단×(1+target) 지정가 — 쿼터매도.
    - 전량 매도 시 사이클 종료 → 다음 거래일 실현손익 재투자(복리)로
      새 사이클 시작.
    - 40분할 소진 시 ``depletion_mode``: "hold"(매수 중단, 매도 대기 —
      기본) / "stop_loss"(전량 종가 손절 후 다음 날 재시작).
    - 체결 판정: 지정가 매도는 인트라데이(15m) 데이터가 있으면 봉
      단위로(갭 오픈 포함), 없으면 일봉 시가/고가 근사.
    """

    ticker: str = "TQQQ"
    start_date: date
    end_date: date
    initial_capital: float = 100_000_000.0
    divisions: int = 40
    version: str = "v2.2"  # "v2.1" | "v2.2"
    target_profit_pct: float = 0.10
    depletion_mode: str = "hold"  # "hold" | "stop_loss"
    fee_schedule: TossFeeSchedule = Field(default_factory=TossFeeSchedule)
    capital_gains_tax_pct: float = 0.22
    tax_deduction: float = 2_500_000.0


class InfiniteBuyingEvent(BaseModel):
    """무한매수 매매 이벤트.

    ``kind``: "start_buy"(사이클 첫 1T) / "big_buy"(큰수 LOC) /
    "avg_buy"(평단 LOC) / "quarter_sell"(v2.2 쿼터 LOC 매도) /
    "limit_sell"(지정가 익절) / "stop_loss"(소진 손절) /
    "tax"(연초 정산). ``intraday``는 15m 데이터로 체결된 경우 True.

    매수 한 건 한 건의 맥락 재구성용 필드:
    ``qty``(거래 수량), ``tranche_no``(사이클 내 몇 번째 매수 —
    매도/세금은 0), ``spent_pct``(체결 후 사이클 원금 투입률),
    ``target_price``(체결 후 평단 기준 익절 목표가).
    """

    ts: datetime
    kind: str
    price: float
    notional: float
    qty: float = 0.0
    shares_after: float
    avg_price_after: float
    cash_after: float
    cycle_no: int
    tranche_no: int = 0
    spent_pct: float = 0.0
    target_price: float = 0.0
    intraday: bool = False


class InfiniteBuyingCycle(BaseModel):
    """한 사이클(진입→전량 청산) 기록."""

    cycle_no: int
    start: date
    end: date | None
    trading_days: int
    invested_max: float
    pnl: float
    pnl_pct: float
    outcome: str  # "profit" | "stop_loss" | "open"(진행 중)
    depleted: bool


class InfiniteBuyingPoint(BaseModel):
    date: date
    total: float
    stock_value: float
    cash: float
    avg_price: float
    tranches_spent_pct: float  # 사이클 원금 대비 투입 비율 (0~1)


class InfiniteBuyingResult(BaseModel):
    config: InfiniteBuyingConfig
    curve: list[InfiniteBuyingPoint]
    events: list[InfiniteBuyingEvent]
    cycles: list[InfiniteBuyingCycle]
    summary: PortfolioSummary
    liquidation: LiquidationSummary
    benchmarks: list[PortfolioSummary]
    benchmark_curves: dict[str, list[EquityPoint]]
    benchmark_after_tax: dict[str, float]
    intraday_days: int  # 15m 체결 판정이 적용된 거래일 수


class MultiStrategyResult(BaseModel):
    config: MultiStrategyConfig
    tickers_scanned: int
    total_signals: int
    trades_taken: int
    win_rate: float
    total_return_pct: float
    initial_capital: float
    final_capital: float
    max_drawdown_pct: float
    total_commission: float
    trades: list[MultiTrade]
    equity_curve: list[EquityPoint]
    failed_tickers: list[str]
