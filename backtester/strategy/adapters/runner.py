from backtest.domain.models import BacktestResult
from backtest.domain.ports import BacktestEnginePort
from data.domain.ports import MarketDataPort
from pattern.domain.ports import PatternDetector
from strategy.domain.models import EquityPoint, StrategyConfig, StrategyResult
from strategy.domain.ports import StrategyRunnerPort


class StrategyRunner(StrategyRunnerPort):
    """Orchestrates data fetch -> pattern detection -> backtesting."""

    def __init__(
        self,
        market_data: MarketDataPort,
        detectors: dict[str, PatternDetector],
        engine: BacktestEnginePort,
    ):
        self._market_data = market_data
        self._detectors = detectors
        self._engine = engine

    def run(self, config: StrategyConfig) -> StrategyResult:
        df = self._market_data.fetch_ohlcv(
            config.ticker, config.start_date, config.end_date
        )

        detector = self._detectors[config.pattern_name]
        signals = detector.detect(df)

        backtest_result = self._engine.run(df, signals)
        equity_curve = self._build_equity_curve(backtest_result, config)

        return StrategyResult(
            config=config,
            backtest_result=backtest_result,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _build_equity_curve(
        result: BacktestResult,
        config: StrategyConfig,
    ) -> list[EquityPoint]:
        equity = config.initial_capital
        curve = [EquityPoint(date=config.start_date, equity=equity)]

        for trade in result.trades:
            equity += trade.pnl
            curve.append(EquityPoint(date=trade.exit_date, equity=round(equity, 2)))

        return curve
