from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import pandas as pd

from data.domain.ports import MarketDataPort, UniverseProviderPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from strategy.adapters.wickplay_strategy import WickPlayStrategy
from strategy.domain.models import (
    EquityPoint,
    MultiStrategyConfig,
    MultiStrategyResult,
    MultiTrade,
    StrategyConfig,
)


class MultiWickPlayStrategy:
    """Single-portfolio Wick Play strategy across an entire universe.

    Mirror of :class:`MultiWedgepopStrategy` but tailored to Wick Play:

      - Per-ticker trading logic is delegated to :class:`WickPlayStrategy`
        so entry/exit/sizing stay identical to the single-ticker page.
      - The ``volume_ratio < 1.0`` hard filter used by Multi-Wedgepop is
        **dropped** — Wick Play's own 4-check psychology score already
        gates low-conviction signals at the detector level.
      - Daily auction still picks the signal with the highest
        **buy/sell volume ratio** (Accumulation/Distribution proxy)
        when multiple tickers fire the same day.

    Two-phase execution

        Phase 1 — **scan** (parallel, I/O bound)
            For each ticker: fetch OHLCV → decorate indicators → run
            the Wick Play detector. Output: ``{ticker: (df, signals,
            exit_dates)}``.

        Phase 2 — **walk** (sequential, deterministic)
            Walk signal dates chronologically. While flat, pick the
            signal with the highest buy/sell ratio and delegate
            entry/exit to :class:`WickPlayStrategy._execute_trade`.
            While a position is open, every other signal is ignored
            until exit.

    Position sizing uses ``risk_per_trade × current capital``
    (compound), so winning trades grow the next position naturally.
    """

    def __init__(
        self,
        market_data: MarketDataPort,
        universe_provider: UniverseProviderPort,
        detector: PatternDetector,
        strategy: WickPlayStrategy | None = None,
        max_workers: int = 8,
        min_bars: int = 15,
    ):
        self._market_data = market_data
        self._universe_provider = universe_provider
        self._detector = detector
        self._strategy = strategy or WickPlayStrategy(
            market_data=market_data, detector=detector
        )
        self._max_workers = max_workers
        self._min_bars = min_bars

    # ---- public API ----

    def run(self, config: MultiStrategyConfig) -> MultiStrategyResult:
        tickers = self._universe_provider.get_tickers(config.universe)
        if config.max_tickers is not None:
            tickers = tickers[: config.max_tickers]

        ticker_state, failed = self._scan_universe(tickers, config)
        signals_by_date, total_signals = self._collect_signals(
            ticker_state, config
        )
        trades, equity_curve, final_capital, max_dd = self._walk_signals(
            signals_by_date, ticker_state, config
        )

        return self._build_result(
            config=config,
            tickers_scanned=len(tickers),
            total_signals=total_signals,
            trades=trades,
            equity_curve=equity_curve,
            final_capital=final_capital,
            max_dd=max_dd,
            failed=failed,
        )

    # ---- phase 1: scan ----

    def _scan_universe(
        self, tickers: list[str], config: MultiStrategyConfig
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        ticker_state: dict[str, dict[str, Any]] = {}
        failed: list[str] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {
                ex.submit(self._scan_ticker, t, config): t for t in tickers
            }
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    state = fut.result()
                except Exception:
                    failed.append(t)
                    continue
                if state is None:
                    failed.append(t)
                    continue
                ticker_state[t] = state

        return ticker_state, failed

    def _scan_ticker(
        self, ticker: str, config: MultiStrategyConfig
    ) -> dict[str, Any] | None:
        # Fetch extra history so EMA / psych vol average converge from
        # day 1 of the user's window.
        fetch_start = config.start_date - timedelta(days=150)
        try:
            df = self._market_data.fetch_ohlcv(
                ticker, fetch_start, config.end_date
            )
        except Exception:
            return None
        if df is None or df.empty or len(df) < self._min_bars:
            return None

        df_ind = self._strategy._with_indicators(df)
        signals = self._detector.detect(df_ind)

        # Pre-compute pattern-based exit dates for this ticker when
        # the injected strategy has an Exhaustion exit detector. Safe
        # against look-ahead — all detector conditions are end-of-bar.
        exit_dates: set[date] = set()
        exit_detector = getattr(self._strategy, "exit_detector", None)
        if exit_detector is not None:
            exit_dates = {s.date for s in exit_detector.detect(df_ind)}

        return {
            "df": df_ind,
            "signals": signals,
            "exit_dates": exit_dates,
        }

    # ---- phase 2a: collect signals ----

    def _collect_signals(
        self,
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[dict[date, list[dict[str, Any]]], int]:
        """Collect all signals tagged with their buy/sell pressure.

        Signals outside ``[config.start_date, config.end_date]`` are
        silently discarded (they come from the warmup bars). No extra
        volume filter beyond what the detector's psychology score
        already enforces.
        """
        by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
        total = 0
        for ticker, state in ticker_state.items():
            df: pd.DataFrame = state["df"]
            for signal in state["signals"]:
                if signal.date < config.start_date or signal.date > config.end_date:
                    continue
                pressure = self._signal_pressure(df, signal)
                if pressure is None:
                    continue
                by_date[signal.date].append(
                    {"ticker": ticker, "signal": signal, **pressure}
                )
                total += 1
        return by_date, total

    @staticmethod
    def _signal_pressure(
        df: pd.DataFrame, signal: PatternSignal
    ) -> dict[str, float] | None:
        """Estimate buy vs sell volume on the signal day from OHLC.

        Same Accumulation/Distribution proxy used by MultiWedgepop:
        splits bar volume by close-within-range position. Returns
        None when the signal date isn't in the index.
        """
        ts = pd.Timestamp(signal.date)
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        if ts not in df.index:
            return None

        row = df.loc[ts]
        high = float(row["High"])
        low = float(row["Low"])
        close = float(row["Close"])
        volume = float(row["Volume"])
        bar_range = high - low

        if bar_range <= 0 or volume <= 0:
            half = volume / 2.0
            return {
                "volume": volume,
                "buy_volume": half,
                "sell_volume": half,
                "buy_sell_ratio": 1.0,
            }

        buy_vol = (close - low) / bar_range * volume
        sell_vol = (high - close) / bar_range * volume
        ratio = buy_vol / (sell_vol + 1.0)
        return {
            "volume": volume,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "buy_sell_ratio": ratio,
        }

    # ---- phase 2b: walk signals ----

    def _walk_signals(
        self,
        signals_by_date: dict[date, list[dict[str, Any]]],
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[list[MultiTrade], list[EquityPoint], float, float]:
        capital = config.initial_capital
        peak = capital
        max_dd = 0.0
        trades: list[MultiTrade] = []
        curve: list[EquityPoint] = [
            EquityPoint(date=config.start_date, equity=capital)
        ]
        free_after_date: date | None = None

        for signal_date in sorted(signals_by_date.keys()):
            if free_after_date is not None and signal_date <= free_after_date:
                continue

            best = max(
                signals_by_date[signal_date],
                key=lambda c: c["buy_sell_ratio"],
            )
            ticker = best["ticker"]
            df = ticker_state[ticker]["df"]
            exit_dates = ticker_state[ticker].get("exit_dates", set())
            signal: PatternSignal = best["signal"]

            multi_trade, exit_date = self._execute_one(
                ticker=ticker,
                df=df,
                signal=signal,
                pressure=best,
                capital=capital,
                config=config,
                exit_dates=exit_dates,
            )
            if multi_trade is None or exit_date is None:
                continue

            capital += multi_trade.pnl
            peak = max(peak, capital)
            if peak > 0:
                max_dd = max(max_dd, (peak - capital) / peak)
            trades.append(multi_trade)
            curve.append(
                EquityPoint(date=exit_date, equity=round(capital, 2))
            )
            free_after_date = exit_date

        return trades, curve, capital, max_dd

    def _execute_one(
        self,
        ticker: str,
        df: pd.DataFrame,
        signal: PatternSignal,
        pressure: dict[str, float],
        capital: float,
        config: MultiStrategyConfig,
        exit_dates: set[date] | None = None,
    ) -> tuple[MultiTrade | None, date | None]:
        entry_idx = self._strategy._next_open_index(df, signal.date)
        if entry_idx is None:
            return None, None

        per_config = StrategyConfig(
            ticker=ticker,
            start_date=config.start_date,
            end_date=config.end_date,
            pattern_name=config.pattern_name,
            initial_capital=capital,
            risk_per_trade=config.risk_per_trade,
            max_holding_days=config.max_holding_days,
        )
        trade, _ = self._strategy._execute_trade(
            df, signal, entry_idx, capital, per_config,
            exit_dates or set(),
        )
        if trade is None:
            return None, None

        commission = config.fee_schedule.round_trip(
            trade.entry_price, trade.exit_price, trade.shares
        )
        gross_pnl = trade.pnl
        net_pnl = round(gross_pnl - commission, 2)
        cost_basis = trade.entry_price * trade.shares
        net_pnl_pct = (
            round(net_pnl / cost_basis, 4) if cost_basis > 0 else 0.0
        )

        multi_trade = MultiTrade(
            ticker=ticker,
            signal_volume=pressure["volume"],
            signal_buy_volume=round(pressure["buy_volume"], 2),
            signal_sell_volume=round(pressure["sell_volume"], 2),
            signal_buy_sell_ratio=round(pressure["buy_sell_ratio"], 4),
            commission=round(commission, 2),
            gross_pnl=round(gross_pnl, 2),
            pattern_name=trade.pattern_name,
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            stop_loss=trade.stop_loss,
            shares=trade.shares,
            pnl=net_pnl,
            pnl_pct=net_pnl_pct,
            exit_reason=trade.exit_reason,
        )
        return multi_trade, trade.exit_date

    # ---- result ----

    @staticmethod
    def _build_result(
        config: MultiStrategyConfig,
        tickers_scanned: int,
        total_signals: int,
        trades: list[MultiTrade],
        equity_curve: list[EquityPoint],
        final_capital: float,
        max_dd: float,
        failed: list[str],
    ) -> MultiStrategyResult:
        wins = [t for t in trades if t.pnl > 0]
        win_rate = round(len(wins) / len(trades), 4) if trades else 0.0
        total_return = (
            round(
                (final_capital - config.initial_capital)
                / config.initial_capital,
                4,
            )
            if config.initial_capital > 0
            else 0.0
        )
        total_commission = round(sum(t.commission for t in trades), 2)

        return MultiStrategyResult(
            config=config,
            tickers_scanned=tickers_scanned,
            total_signals=total_signals,
            trades_taken=len(trades),
            win_rate=win_rate,
            total_return_pct=total_return,
            initial_capital=config.initial_capital,
            final_capital=round(final_capital, 2),
            max_drawdown_pct=round(max_dd, 4),
            total_commission=total_commission,
            trades=trades,
            equity_curve=equity_curve,
            failed_tickers=failed,
        )
