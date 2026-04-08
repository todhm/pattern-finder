from datetime import date

import pandas as pd

from backtest.adapters.engine import BacktestEngine
from pattern.adapters.reversal_extension import ReversalExtensionDetector
from pattern.adapters.wedge_pop import WedgePopDetector


class TestBacktestEngine:
    def setup_method(self):
        self.engine = BacktestEngine(
            initial_capital=100_000,
            risk_per_trade=0.02,
            max_holding_days=30,
        )

    def test_run_with_reversal_signals(self, downtrend_with_reversal: pd.DataFrame):
        detector = ReversalExtensionDetector(
            extension_pct=0.03, volume_multiplier=1.5, lookback=5
        )
        signals = detector.detect(downtrend_with_reversal)
        result = self.engine.run(downtrend_with_reversal, signals)

        assert result.initial_capital == 100_000
        assert result.total_trades == len(result.trades)
        assert 0.0 <= result.win_rate <= 1.0

    def test_run_with_wedge_pop_signals(
        self, consolidation_with_breakout: pd.DataFrame
    ):
        detector = WedgePopDetector()
        signals = detector.detect(consolidation_with_breakout)
        result = self.engine.run(consolidation_with_breakout, signals)

        assert result.initial_capital == 100_000
        assert result.total_trades == len(result.trades)

    def test_empty_signals(self, flat_market: pd.DataFrame):
        result = self.engine.run(flat_market, [])
        assert result.total_trades == 0
        assert result.final_capital == 100_000
        assert result.win_rate == 0.0

    def test_trade_fields(self, downtrend_with_reversal: pd.DataFrame):
        detector = ReversalExtensionDetector(
            extension_pct=0.03, volume_multiplier=1.5, lookback=5
        )
        signals = detector.detect(downtrend_with_reversal)
        result = self.engine.run(downtrend_with_reversal, signals)

        for trade in result.trades:
            assert trade.shares >= 1
            assert trade.entry_date <= trade.exit_date
            assert trade.stop_loss < trade.entry_price

    def test_max_drawdown_non_negative(self, downtrend_with_reversal: pd.DataFrame):
        detector = ReversalExtensionDetector(
            extension_pct=0.03, volume_multiplier=1.5, lookback=5
        )
        signals = detector.detect(downtrend_with_reversal)
        result = self.engine.run(downtrend_with_reversal, signals)
        assert result.max_drawdown_pct >= 0.0
