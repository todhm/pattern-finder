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

    @property
    def dip_buy_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "dip_buy")

    @property
    def profit_take_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "profit_take")


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
