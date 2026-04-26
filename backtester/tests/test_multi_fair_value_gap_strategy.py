"""MultiFairValueGapStrategy unit tests.

Reuses the standard intraday CHoCH+FVG fixture from
``test_fair_value_gap.py`` to verify the universe runner correctly
delegates per-ticker entry/exit to :class:`FairValueGapStrategy` and
produces a populated :class:`MultiStrategyResult`.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from data.adapters.wikipedia_universe import StaticUniverseAdapter
from data.domain.ports import MarketDataPort
from pattern.adapters.fair_value_gap import FairValueGapDetector
from strategy.adapters.fair_value_gap_strategy import FairValueGapStrategy
from strategy.adapters.multi_fair_value_gap_strategy import (
    MultiFairValueGapStrategy,
)
from strategy.domain.models import MultiStrategyConfig, TossFeeSchedule
from tests.test_fair_value_gap import _build_choch_fvg_session


ZERO_FEES = TossFeeSchedule(
    buy_commission_pct=0.0, sell_commission_pct=0.0, sec_fee_pct=0.0
)


class _FakeUniverseMarketData(MarketDataPort):
    """Per-ticker DataFrame map. Raises KeyError for unknown tickers
    so the runner's ``failed`` bookkeeping is exercised."""

    def __init__(
        self,
        tables: dict[str, pd.DataFrame],
        failing: set[str] | None = None,
    ) -> None:
        self._tables = tables
        self._failing = failing or set()

    def fetch_ohlcv(self, symbol, start, end, interval="1d"):
        if symbol in self._failing:
            raise RuntimeError(f"simulated fetch failure for {symbol}")
        if symbol not in self._tables:
            raise KeyError(symbol)
        return self._tables[symbol]


def _build_runner(
    tables: dict[str, pd.DataFrame],
    *,
    failing: set[str] | None = None,
    take_profit_r: float = 3.0,
    enable_breakeven: bool = False,
    enable_bos_trail: bool = False,
) -> MultiFairValueGapStrategy:
    market_data = _FakeUniverseMarketData(tables, failing=failing)
    universe = StaticUniverseAdapter({"test": list(tables.keys())})
    detector = FairValueGapDetector(
        min_gap_pct=0.0, max_signals_per_session=1
    )
    per_ticker = FairValueGapStrategy(
        market_data=market_data,
        detector=detector,
        take_profit_r_multiple=take_profit_r,
        enable_breakeven_stop=enable_breakeven,
        enable_bos_trail=enable_bos_trail,
    )
    return MultiFairValueGapStrategy(
        market_data=market_data,
        universe_provider=universe,
        detector=detector,
        strategy=per_ticker,
        max_workers=2,
        min_bars=5,
    )


def _config(tables, **overrides) -> MultiStrategyConfig:
    # Use a window covering the fixture's 2024-03-18 session.
    base = dict(
        universe="test",
        start_date=date(2024, 3, 18),
        end_date=date(2024, 3, 18),
        pattern_name="fair_value_gap",
        initial_capital=100_000.0,
        risk_per_trade=0.02,
        max_holding_days=20,
        max_tickers=len(tables),
        fee_schedule=ZERO_FEES,
    )
    base.update(overrides)
    return MultiStrategyConfig(**base)


def test_multi_runner_executes_single_ticker_signal_to_take_profit() -> None:
    """One ticker, one signal — should result in 1 trade exiting at
    the 3R take-profit target. Same fixture / numbers as the
    single-ticker ``test_strategy_take_profit_fires_at_3r``."""
    df = _build_choch_fvg_session()
    runner = _build_runner({"AAPL": df})
    result = runner.run(_config({"AAPL": df}))

    assert result.tickers_scanned == 1
    assert result.total_signals == 1
    assert result.trades_taken == 1
    assert result.failed_tickers == []

    trade = result.trades[0]
    assert trade.ticker == "AAPL"
    assert trade.exit_reason == "take_profit"
    assert trade.entry_price == pytest.approx(107.0)  # fvg_mid
    assert trade.exit_price == pytest.approx(122.0)   # 107 + 3R(=5)
    assert trade.pnl > 0
    # Equity curve has the start anchor + one exit point.
    assert len(result.equity_curve) == 2
    assert result.final_capital > result.initial_capital


def test_multi_runner_picks_higher_buy_pressure_on_concurrent_signals() -> None:
    """Two tickers print FVG retest on the same bar. The runner must
    take the one whose retest bar shows stronger buy-side A/D
    pressure (close pinned near the top of the bar's range)."""
    df_strong_buy = _build_choch_fvg_session().copy().astype(float)
    df_weak_buy = _build_choch_fvg_session().copy().astype(float)
    # Bar 19 is the retest bar in the standard fixture. Tweak its
    # close-within-range so AAPL's signal looks like an aggressive
    # buy (close near high) and MSFT's looks like a tepid buy.
    # Detector retest condition: close > mid (107), low <= mid,
    # low >= fvg_low (104). Both bars satisfy that — only A/D
    # buy/sell ratio differs.
    bar19_idx = 19
    high_col = df_strong_buy.columns.get_loc("High")
    low_col = df_strong_buy.columns.get_loc("Low")
    close_col = df_strong_buy.columns.get_loc("Close")
    df_strong_buy.iat[bar19_idx, high_col] = 113.0
    df_strong_buy.iat[bar19_idx, low_col] = 105.0
    df_strong_buy.iat[bar19_idx, close_col] = 112.5
    df_weak_buy.iat[bar19_idx, high_col] = 113.0
    df_weak_buy.iat[bar19_idx, low_col] = 105.0
    df_weak_buy.iat[bar19_idx, close_col] = 107.5

    runner = _build_runner(
        {"AAPL": df_strong_buy, "MSFT": df_weak_buy}
    )
    cfg = _config({"AAPL": df_strong_buy, "MSFT": df_weak_buy})
    result = runner.run(cfg)

    # Both tickers have a signal at the same bar but only one
    # position at a time — the higher buy/sell ratio (AAPL) wins.
    assert result.total_signals == 2
    assert result.trades_taken == 1
    assert result.trades[0].ticker == "AAPL"


def test_multi_runner_records_failed_fetches() -> None:
    """A ticker whose fetch raises is captured in ``failed_tickers``
    and doesn't break the rest of the scan."""
    df = _build_choch_fvg_session()
    runner = _build_runner({"AAPL": df, "BAD": df}, failing={"BAD"})
    result = runner.run(_config({"AAPL": df, "BAD": df}))

    assert "BAD" in result.failed_tickers
    assert result.tickers_scanned == 2
    # AAPL still processed normally.
    assert result.trades_taken == 1
    assert result.trades[0].ticker == "AAPL"


def test_multi_runner_applies_round_trip_commission() -> None:
    """With non-zero fee schedule, ``MultiTrade.commission`` is
    populated and ``pnl`` is reduced by the same amount."""
    df = _build_choch_fvg_session()
    runner = _build_runner({"AAPL": df})
    nonzero_fees = TossFeeSchedule(
        buy_commission_pct=0.001,  # 0.1%
        sell_commission_pct=0.001,
        sec_fee_pct=0.0,
    )
    cfg = _config({"AAPL": df}, fee_schedule=nonzero_fees)
    result = runner.run(cfg)
    assert result.trades_taken == 1
    trade = result.trades[0]
    assert trade.commission > 0
    # gross_pnl is recorded separately, net pnl = gross - commission.
    assert trade.pnl == pytest.approx(trade.gross_pnl - trade.commission)
    assert result.total_commission == trade.commission
