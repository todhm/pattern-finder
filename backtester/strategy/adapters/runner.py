from backtest.domain.models import BacktestResult
from backtest.domain.ports import BacktestEnginePort
from data.domain.ports import MarketDataPort
from pattern.domain.ports import PatternDetector
from strategy.domain.models import (
    EquityPoint,
    StrategyConfig,
    StrategyPerformance,
    StrategyResult,
    Trade,
)
from strategy.domain.ports import StrategyRunnerPort


class StrategyRunner(StrategyRunnerPort):
    """Bridge adapter: legacy `BacktestEnginePort` -> strategy domain.

    The strategy domain is intentionally free of any backtest types. This
    runner sits at the adapter boundary: it delegates trade execution to
    the legacy `BacktestEngine` and translates the result into the
    strategy-owned types (`StrategyPerformance`, `Trade`). All knowledge
    of the backtest domain is contained in this single file.

    For new patterns, prefer writing a self-contained strategy adapter
    (e.g. `WedgepopStrategy`) instead of going through this bridge.
    """

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
        performance = self._to_performance(backtest_result)
        equity_curve = self._build_equity_curve(performance, config)

        return StrategyResult(
            config=config,
            performance=performance,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _to_performance(br: BacktestResult) -> StrategyPerformance:
        return StrategyPerformance(
            initial_capital=br.initial_capital,
            final_capital=br.final_capital,
            total_return_pct=br.total_return_pct,
            total_trades=br.total_trades,
            win_rate=br.win_rate,
            avg_win_pct=br.avg_win_pct,
            avg_loss_pct=br.avg_loss_pct,
            max_drawdown_pct=br.max_drawdown_pct,
            trades=[
                Trade(
                    pattern_name=t.pattern_name,
                    entry_date=t.entry_date,
                    exit_date=t.exit_date,
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    stop_loss=t.stop_loss,
                    shares=t.shares,
                    pnl=t.pnl,
                    pnl_pct=t.pnl_pct,
                )
                for t in br.trades
            ],
        )

    @staticmethod
    def _build_equity_curve(
        performance: StrategyPerformance,
        config: StrategyConfig,
    ) -> list[EquityPoint]:
        equity = config.initial_capital
        curve = [EquityPoint(date=config.start_date, equity=equity)]

        for trade in performance.trades:
            equity += trade.pnl
            curve.append(EquityPoint(date=trade.exit_date, equity=round(equity, 2)))

        return curve
