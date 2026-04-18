from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any

import pandas as pd

from data.domain.ports import MarketDataPort, UniverseProviderPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from strategy.adapters.wedgepop_strategy import WedgepopStrategy
from strategy.domain.models import (
    EquityPoint,
    MultiStrategyConfig,
    MultiStrategyResult,
    MultiTrade,
    StrategyConfig,
)


class MultiWedgepopStrategy:
    """Single-portfolio Wedge Pop strategy across an entire universe.

    Two-phase execution

        Phase 1 — **scan** (parallel, I/O bound)
            For every ticker in the universe, fetch OHLCV, decorate with
            indicators, and run the wedge-pop detector. The output is a
            map ``ticker -> (df, signals)`` and a flat list of every
            signal tagged with its ticker and the bar's volume.

        Phase 2 — **walk** (sequential, deterministic)
            Walk every signal date in chronological order. While the
            portfolio is flat, the signal with the highest **buy/sell
            volume ratio** wins; we delegate entry/exit/sizing to the
            injected :class:`WedgepopStrategy` so the trading logic
            stays identical to the single-ticker version. While a
            position is open, every other signal — same ticker or
            different — is ignored until the position is closed.

    Selection metric — buy/sell volume ratio

        Raw volume is misleading: a $10B name will dwarf a $500M name
        on volume even if the small one had a much cleaner buying
        bar. Instead we estimate the *intrabar* buy/sell split using
        the close's location within the bar's range — the standard
        Accumulation/Distribution proxy::

            buy_share  = (close - low)  / (high - low)
            sell_share = (high - close) / (high - low)
            buy_vol    = buy_share  × volume
            sell_vol   = sell_share × volume
            ratio      = buy_vol / (sell_vol + 1.0)   # additive smoothing

        Higher ratio = the bar closed nearer the high, which is the
        market's vote that the day's flow was predominantly buying.
        The ``+1.0`` floor on the denominator avoids division by zero
        on the perfect close-on-high case and is negligible against
        equity volumes that are typically in the millions.

    Position sizing uses ``risk_per_trade × current capital``
    (compound), so winning trades grow the next position naturally.
    """

    def __init__(
        self,
        market_data: MarketDataPort,
        universe_provider: UniverseProviderPort,
        detector: PatternDetector,
        strategy: WedgepopStrategy | None = None,
        max_workers: int = 8,
        min_bars: int = 15,
    ):
        self._market_data = market_data
        self._universe_provider = universe_provider
        self._detector = detector
        # `strategy` lets callers inject a tuned WedgepopStrategy
        # (different exit knobs, gap-up filter, etc). When omitted we
        # build a default one — `market_data` is unused on the helper
        # path because we drive `_with_indicators` / `_execute_trade`
        # directly with pre-fetched DataFrames.
        self._strategy = strategy or WedgepopStrategy(
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
        # Fetch extra history so 50/200 SMA converge from day 1.
        from datetime import timedelta

        fetch_start = config.start_date - timedelta(days=400)
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
        # the injected wedgepop strategy has an exit detector wired
        # up. Safe against look-ahead: the detector's conditions are
        # all end-of-bar, and pandas EWM/rolling indicators are
        # causal — each signal's date corresponds to a bar whose
        # detection used only data up to and including that bar.
        exit_dates: set[date] = set()
        exit_detector = getattr(self._strategy, "_exit_detector", None)
        if exit_detector is not None:
            exit_dates = {s.date for s in exit_detector.detect(df_ind)}

        return {
            "df": df_ind,
            "signals": signals,
            "exit_dates": exit_dates,
        }

    # ---- phase 2a: collect signals across universe ----

    def _collect_signals(
        self,
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[dict[date, list[dict[str, Any]]], int]:
        """Collect all signals tagged with their buy/sell pressure.

        Signals outside ``[config.start_date, config.end_date]`` are
        silently discarded (they come from the SMA-warmup bars).

        Hard volume filter: a wedge-pop signal whose breakout-bar
        volume is *below* the 20-day average (``metadata.volume_ratio
        < 1.0``) is excluded from the multi-strategy's candidate
        pool entirely.
        """
        by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
        total = 0
        for ticker, state in ticker_state.items():
            df: pd.DataFrame = state["df"]
            for signal in state["signals"]:
                if signal.date < config.start_date or signal.date > config.end_date:
                    continue
                vol_ratio = signal.metadata.get("volume_ratio", 1.0)
                if vol_ratio < 1.0:
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

        Splits the bar's volume into buy/sell shares using the close's
        location within the high–low range — the Accumulation /
        Distribution proxy. Returns None when the signal date is not
        in the index. Falls back to a neutral 1:1 split for degenerate
        bars (high == low or zero volume).
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
        # Additive smoothing prevents division by zero when close ==
        # high (perfect close-on-high). The +1.0 floor is negligible
        # vs typical equity volumes (millions).
        ratio = buy_vol / (sell_vol + 1.0)
        return {
            "volume": volume,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "buy_sell_ratio": ratio,
        }

    # ---- phase 2b: walk signals chronologically ----

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
        trade, exit_idx = self._strategy._execute_trade(
            df, signal, entry_idx, capital, per_config,
            exit_dates or set(),
        )
        if trade is None:
            return None, None

        # Apply Toss commission. The single-ticker strategy returns a
        # gross trade; we deduct the round-trip fee here so the
        # multi-strategy's pnl / pnl_pct reflect what the user would
        # actually take home through Toss Securities.
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
