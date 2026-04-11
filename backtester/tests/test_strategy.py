from datetime import date

import pandas as pd

from backtest.adapters.engine import BacktestEngine
from data.domain.ports import MarketDataPort
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_pop import WedgePopDetector
from strategy.adapters.runner import StrategyRunner
from strategy.domain.models import StrategyConfig


class FakeMarketData(MarketDataPort):
    """Returns a pre-built DataFrame instead of calling Yahoo Finance."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def fetch_ohlcv(self, symbol, start, end):
        return self._df


class TestStrategyRunner:
    def _build_runner(self, df: pd.DataFrame) -> StrategyRunner:
        return StrategyRunner(
            market_data=FakeMarketData(df),
            detectors={
                "reversal_extension": ReversalExtensionDetector(
                    extension_pct=0.03, volume_multiplier=1.5, lookback=5
                ),
                "wedge_pop": WedgePopDetector(),
            },
            engine=BacktestEngine(initial_capital=100_000, risk_per_trade=0.02),
        )

    def test_run_reversal_strategy(self, downtrend_with_reversal: pd.DataFrame):
        runner = self._build_runner(downtrend_with_reversal)
        config = StrategyConfig(
            ticker="TEST",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            pattern_name="reversal_extension",
        )
        result = runner.run(config)

        assert result.config == config
        assert result.performance.initial_capital == 100_000
        assert len(result.equity_curve) >= 1

    def test_run_wedge_pop_strategy(self, consolidation_with_breakout: pd.DataFrame):
        runner = self._build_runner(consolidation_with_breakout)
        config = StrategyConfig(
            ticker="TEST",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            pattern_name="wedge_pop",
        )
        result = runner.run(config)

        assert result.config == config
        assert result.performance.total_trades == len(
            result.performance.trades
        )

    def test_equity_curve_starts_with_initial_capital(
        self, downtrend_with_reversal: pd.DataFrame
    ):
        runner = self._build_runner(downtrend_with_reversal)
        config = StrategyConfig(
            ticker="TEST",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            pattern_name="reversal_extension",
        )
        result = runner.run(config)
        assert result.equity_curve[0].equity == 100_000

    def test_no_signals_produces_flat_curve(self, flat_market: pd.DataFrame):
        runner = self._build_runner(flat_market)
        config = StrategyConfig(
            ticker="TEST",
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            pattern_name="reversal_extension",
        )
        result = runner.run(config)
        assert result.performance.total_trades == 0
        assert len(result.equity_curve) == 1
        assert result.equity_curve[0].equity == 100_000
