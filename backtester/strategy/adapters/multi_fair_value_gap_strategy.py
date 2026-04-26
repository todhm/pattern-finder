"""Universe-wide Fair Value Gap strategy.

Single-portfolio runner that scans the S&P 500 / Nasdaq-100 (or any
``UniverseProviderPort``-supplied list) for FVG signals and walks
them chronologically, keeping at most one position open at a time.

Lifecycle mirrors :class:`MultiWickPlayStrategy`:

    Phase 1 — **scan** (parallel, I/O bound)
        For each ticker: fetch intraday OHLCV → run the
        :class:`FairValueGapDetector`. Output keyed on the ticker
        symbol so phase 2 can index back into the per-ticker df for
        the per-trade exit walk.

    Phase 2a — **collect**
        Group signals by their bar timestamp (intraday-precise — two
        15m signals on the same NY date stay in separate buckets).
        Each signal carries an A/D-style buy/sell-volume estimate
        derived from the bar's close-within-range position so phase
        2b can pick the highest-conviction one when multiple
        tickers fire on the same bar.

    Phase 2b — **walk** (sequential, deterministic)
        For each signal-key in chronological order: while flat, take
        the highest-buy/sell-ratio signal in that bucket and delegate
        entry/sizing/exit to :class:`FairValueGapStrategy`. While a
        position is open, *every* later signal is skipped until the
        existing trade exits — the runner does not stack positions.

The whole orchestration is intentionally distinct from the wedgepop /
wickplay multi runners because :class:`FairValueGapStrategy` doesn't
expose the bar-identity hooks (``_signal_key``, ``_signal_match_ts``,
``_trade_exit_key``, optional ``exit_detector``) that those use.
Mirroring the algorithm here in ~250 lines is cheaper than retrofitting
those hooks just to share a base class.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

import pandas as pd

from data.domain.ports import MarketDataPort, UniverseProviderPort
from pattern.domain.models import PatternSignal
from pattern.domain.ports import PatternDetector
from strategy.adapters.fair_value_gap_strategy import FairValueGapStrategy
from strategy.domain.models import (
    EquityPoint,
    MultiStrategyConfig,
    MultiStrategyResult,
    MultiTrade,
    StrategyConfig,
)


class MultiFairValueGapStrategy:
    """Single-portfolio FVG strategy across a ticker universe."""

    # 1m interval caps at 7 calendar days on yfinance, 15m / 5m / 30m
    # at 60. ``_max_fetch_lookback_days`` clamps warmup so a user
    # picking a window near the cap boundary still gets a successful
    # fetch instead of a "data not available" silent fail.
    _warmup_days: int = 5
    _max_fetch_lookback_days: int = 58

    def __init__(
        self,
        market_data: MarketDataPort,
        universe_provider: UniverseProviderPort,
        detector: PatternDetector,
        strategy: FairValueGapStrategy | None = None,
        max_workers: int = 4,
        min_bars: int = 30,
    ) -> None:
        self._market_data = market_data
        self._universe_provider = universe_provider
        self._detector = detector
        self._strategy = strategy or FairValueGapStrategy(
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
        signals_by_key, total_signals = self._collect_signals(
            ticker_state, config
        )
        trades, equity_curve, final_capital, max_dd = self._walk_signals(
            signals_by_key, ticker_state, config
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
        # Clamp the start to the upstream's intraday cap so a window
        # set at the boundary still resolves to a valid fetch range.
        floor_start = date.today() - timedelta(
            days=self._max_fetch_lookback_days
        )
        fetch_start = max(
            config.start_date - timedelta(days=self._warmup_days),
            floor_start,
        )
        try:
            df = self._market_data.fetch_ohlcv(
                ticker,
                fetch_start,
                config.end_date,
                interval=self._strategy._interval,
            )
        except Exception:
            return None
        if df is None or df.empty or len(df) < self._min_bars:
            return None
        signals = self._detector.detect(df)
        return {"df": df, "signals": signals}

    # ---- phase 2a: collect ----

    def _collect_signals(
        self,
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[dict[pd.Timestamp, list[dict[str, Any]]], int]:
        """Group signals by bar timestamp. Bars outside the user's
        window are dropped silently — they fell into the warmup
        prefix the scanner overshot."""
        by_key: dict[pd.Timestamp, list[dict[str, Any]]] = defaultdict(list)
        total = 0
        for ticker, state in ticker_state.items():
            df: pd.DataFrame = state["df"]
            for signal in state["signals"]:
                if (
                    signal.date < config.start_date
                    or signal.date > config.end_date
                ):
                    continue
                pressure = self._signal_pressure(df, signal)
                if pressure is None:
                    continue
                key = pd.Timestamp(signal.timestamp)
                by_key[key].append(
                    {"ticker": ticker, "signal": signal, **pressure}
                )
                total += 1
        return by_key, total

    @staticmethod
    def _signal_pressure(
        df: pd.DataFrame, signal: PatternSignal
    ) -> dict[str, float] | None:
        """A/D buy/sell-volume split on the signal bar.

        Same proxy used by the other multi runners: split the bar's
        volume by ``(close-low)/range`` (buy) and ``(high-close)/range``
        (sell). Returns ``None`` when the signal bar is missing from
        the per-ticker df (shouldn't happen in practice but guards
        against tz-mismatches between detector index and stored df).
        """
        ts = pd.Timestamp(signal.timestamp)
        if df.index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(df.index.tz)
        elif df.index.tz is None and ts.tz is not None:
            ts = ts.tz_localize(None)
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

    # ---- phase 2b: walk ----

    def _walk_signals(
        self,
        signals_by_key: dict[pd.Timestamp, list[dict[str, Any]]],
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
        # Block until the previous trade's exit *bar* is past — same
        # within-day re-entry semantics as the 15m / 1m runners.
        free_after_key: pd.Timestamp | None = None

        for signal_key in sorted(signals_by_key.keys()):
            if free_after_key is not None and signal_key <= free_after_key:
                continue
            best = max(
                signals_by_key[signal_key],
                key=lambda c: c["buy_sell_ratio"],
            )
            ticker = best["ticker"]
            df = ticker_state[ticker]["df"]
            signal: PatternSignal = best["signal"]
            multi_trade, exit_key = self._execute_one(
                ticker=ticker,
                df=df,
                signal=signal,
                pressure=best,
                capital=capital,
                config=config,
            )
            if multi_trade is None or exit_key is None:
                continue
            capital += multi_trade.pnl
            peak = max(peak, capital)
            if peak > 0:
                max_dd = max(max_dd, (peak - capital) / peak)
            trades.append(multi_trade)
            curve.append(
                EquityPoint(
                    date=multi_trade.exit_date, equity=round(capital, 2)
                )
            )
            free_after_key = exit_key
        return trades, curve, capital, max_dd

    def _execute_one(
        self,
        ticker: str,
        df: pd.DataFrame,
        signal: PatternSignal,
        pressure: dict[str, float],
        capital: float,
        config: MultiStrategyConfig,
    ) -> tuple[MultiTrade | None, pd.Timestamp | None]:
        entry_idx = self._strategy._signal_bar_index(df, signal)
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
        trade, _exit_idx = self._strategy._execute_trade(
            df, signal, entry_idx, capital, per_config
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
            entry_ts=trade.entry_ts,
            exit_ts=trade.exit_ts,
        )
        exit_key = (
            pd.Timestamp(trade.exit_ts)
            if trade.exit_ts is not None
            else pd.Timestamp(trade.exit_date)
        )
        return multi_trade, exit_key

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
