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
