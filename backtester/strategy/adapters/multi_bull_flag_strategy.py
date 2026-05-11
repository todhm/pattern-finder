"""Universe-wide Bull Flag strategy.

Single-portfolio runner that walks an NY-equity universe day-by-day
and, when multiple tickers fire on the same session, picks the one
with the strongest Ross-style stock-selection profile:

    1차 정렬:  RVOL 내림차순  (오늘의 runner — 거래량이 몰린 쪽이 follow-through ↑)
    2차 정렬:  float 오름차순  (작은 float = 같은 매수 압력에 더 큰 % 이동)
    3차 정렬:  entry_ts 오름차순  (영상 sweet spot = 09:30~10:30 첫 풀백)

Lifecycle (mirrors Multi FCR / Multi Wedgepop):

    Phase 1 — **scan** (parallel)
        For each ticker: fetch 1m + 5m + daily, lookup float via
        yfinance (throttled), run :class:`BullFlagDetector`.
    Phase 2a — **collect**
        Group signals by ``session_date``.
    Phase 2b — **walk** (sequential)
        For each session date: sort candidates by (RVOL desc, float asc,
        entry_ts asc), pick the top one, delegate to
        :class:`BullFlagStrategy` for the single-trade simulation.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as date_t, time, timedelta
from typing import Any

import pandas as pd

from data.domain.market_calendar import MarketCalendar, NY
from data.domain.ports import (
    FundamentalsPort,
    MarketDataPort,
    UniverseProviderPort,
)
from pattern.adapters.bull_flag import BullFlagDetector, BullFlagSignal
from strategy.adapters.bull_flag_strategy import BullFlagStrategy
from strategy.domain.models import (
    EquityPoint,
    MultiStrategyConfig,
    MultiStrategyResult,
    MultiTrade,
    StrategyConfig,
)


class MultiBullFlagStrategy:
    """Single-portfolio Bull Flag strategy across a ticker universe.

    Tiebreaker for same-day signals: ``(RVOL desc, float asc, entry_ts asc)``.
    See module docstring for rationale.
    """

    _warmup_days: int = 5

    def __init__(
        self,
        market_data: MarketDataPort,
        market_data_5m: MarketDataPort,
        daily_market_data: MarketDataPort,
        fundamentals: FundamentalsPort,
        universe_provider: UniverseProviderPort,
        # Detector / strategy params injected via factory args so the
        # page can wire user inputs without re-instantiating the runner.
        detector_factory,
        strategy_factory,
        market: MarketCalendar = NY,
        max_workers: int = 4,
        min_bars: int = 60,
        require_float_filter: bool = True,
    ) -> None:
        self._market_data = market_data
        self._market_data_5m = market_data_5m
        self._daily_market_data = daily_market_data
        self._fundamentals = fundamentals
        self._universe_provider = universe_provider
        # Factories take ``float_shares`` + ``splits`` and return
        # configured detector / strategy. This keeps detector params
        # owned by the page.
        self._detector_factory = detector_factory
        self._strategy_factory = strategy_factory
        self._market = market
        self._max_workers = max_workers
        self._min_bars = min_bars
        self._require_float_filter = require_float_filter

    # ---- public API ----

    def run(self, config: MultiStrategyConfig) -> MultiStrategyResult:
        tickers = self._universe_provider.get_tickers(config.universe)
        if config.max_tickers is not None:
            tickers = tickers[: config.max_tickers]

        ticker_state, failed = self._scan_universe(tickers, config)
        signals_by_date, total_signals = self._collect_signals(
            ticker_state, config
        )
        trades, curve, final_capital, max_dd = self._walk_signals(
            signals_by_date, ticker_state, config
        )
        return self._build_result(
            config=config,
            tickers_scanned=len(tickers),
            total_signals=total_signals,
            trades=trades,
            equity_curve=curve,
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
        self._last_fetch_error: str | None = None
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(self._scan_ticker, t, config): t for t in tickers}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    state = fut.result()
                except Exception as exc:
                    self._last_fetch_error = f"{type(exc).__name__}: {exc}"
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
        # yfinance daily history padded for RVOL / split-blackout calcs.
        daily_start = config.start_date - timedelta(days=120)
        intraday_start = config.start_date - timedelta(days=self._warmup_days)
        try:
            df_intraday = self._market_data.fetch_ohlcv(
                ticker, intraday_start, config.end_date, interval="1m"
            )
            df_5m = self._market_data_5m.fetch_ohlcv(
                ticker, intraday_start, config.end_date, interval="5m"
            )
            df_daily = self._daily_market_data.fetch_ohlcv(
                ticker, daily_start, config.end_date
            )
        except Exception:
            raise
        if df_intraday is None or df_intraday.empty or len(df_intraday) < self._min_bars:
            return None
        if df_daily is None or df_daily.empty:
            return None

        # tz-convert intraday once.
        if (
            df_intraday.index.tz is not None
            and str(df_intraday.index.tz) != self._market.tz
        ):
            df_intraday = df_intraday.copy()
            df_intraday.index = df_intraday.index.tz_convert(self._market.tz)
        if (
            df_5m is not None
            and not df_5m.empty
            and df_5m.index.tz is not None
            and str(df_5m.index.tz) != self._market.tz
        ):
            df_5m = df_5m.copy()
            df_5m.index = df_5m.index.tz_convert(self._market.tz)
        if df_5m is not None and df_5m.empty:
            df_5m = None

        # Float + splits via FundamentalsPort (EODHD primary +
        # Massive fallback + 7-day disk cache). No yfinance rate-limit
        # exposure so this scales to the full NASDAQ universe.
        # Tickers without float data are dropped if require_float_filter
        # — Ross's #1 selection criterion.
        fundamentals = self._fundamentals.fetch(ticker)
        float_shares = fundamentals.float_shares
        splits = fundamentals.splits
        if float_shares is None and self._require_float_filter:
            return None

        detector = self._detector_factory(float_shares=float_shares, splits=splits)
        signals = detector.detect(df_intraday, df_daily, df_5m=df_5m)

        # Daily RVOL by date — used for tiebreaker.
        avg_vol = df_daily["Volume"].rolling(50, min_periods=20).mean()
        drvol = df_daily["Volume"] / avg_vol
        drvol_by_date = {
            ts.date(): float(v)
            for ts, v in drvol.items()
            if pd.notna(v)
        }
        return {
            "df_intraday": df_intraday,
            "df_5m": df_5m,
            "df_daily": df_daily,
            "signals": signals,
            "drvol_by_date": drvol_by_date,
            "float_shares": float_shares,
            "splits": splits,
        }

    # ---- phase 2a: collect ----

    def _collect_signals(
        self,
        ticker_state: dict[str, dict[str, Any]],
        config: MultiStrategyConfig,
    ) -> tuple[dict[date_t, list[dict[str, Any]]], int]:
        by_date: dict[date_t, list[dict[str, Any]]] = defaultdict(list)
        total = 0
        for ticker, state in ticker_state.items():
            drvol_by_date: dict[date_t, float] = state["drvol_by_date"]
            float_shares = state["float_shares"]
            for sig in state["signals"]:
                if (
                    sig.session_date < config.start_date
                    or sig.session_date > config.end_date
                ):
                    continue
                rvol = drvol_by_date.get(sig.session_date, sig.rvol)
                by_date[sig.session_date].append({
                    "ticker": ticker,
                    "signal": sig,
                    "rvol": float(rvol),
                    "float_shares": (
                        float(float_shares) if float_shares else float("inf")
                    ),
                    "entry_ts": pd.Timestamp(sig.entry_ts),
                })
                total += 1
        return by_date, total

    # ---- phase 2b: walk ----

    def _walk_signals(
        self,
        signals_by_date: dict[date_t, list[dict[str, Any]]],
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
        for sess_date in sorted(signals_by_date.keys()):
            # Tiebreaker: RVOL desc → float asc → entry_ts asc.
            ranked = sorted(
                signals_by_date[sess_date],
                key=lambda c: (-c["rvol"], c["float_shares"], c["entry_ts"]),
            )
            best = ranked[0]
            ticker = best["ticker"]
            sig: BullFlagSignal = best["signal"]
            multi_trade = self._execute_one(
                ticker=ticker,
                state=ticker_state[ticker],
                signal=sig,
                fit=best,
                capital=capital,
                config=config,
            )
            if multi_trade is None:
                continue
            capital += multi_trade.pnl
            peak = max(peak, capital)
            if peak > 0:
                max_dd = max(max_dd, (peak - capital) / peak)
            trades.append(multi_trade)
            curve.append(
                EquityPoint(date=multi_trade.exit_date, equity=round(capital, 2))
            )
        return trades, curve, capital, max_dd

    def _execute_one(
        self,
        ticker: str,
        state: dict[str, Any],
        signal: BullFlagSignal,
        fit: dict[str, Any],
        capital: float,
        config: MultiStrategyConfig,
    ) -> MultiTrade | None:
        per_config = StrategyConfig(
            ticker=ticker,
            start_date=config.start_date,
            end_date=config.end_date,
            pattern_name=config.pattern_name,
            initial_capital=capital,
            risk_per_trade=config.risk_per_trade,
            max_holding_days=config.max_holding_days,
        )
        # Build a fresh strategy per ticker — the factory injects the
        # *same* detector that produced this signal so adds / TP /
        # stops behave identically to the single-ticker page.
        detector = self._detector_factory(
            float_shares=state["float_shares"], splits=state["splits"]
        )
        strategy = self._strategy_factory(detector=detector)
        sim_trades, _curve = strategy._simulate(
            df=state["df_intraday"],
            signals=[signal],
            config=per_config,
        )
        if not sim_trades:
            return None
        trade = sim_trades[0]
        # Toss fees already applied inside ``BullFlagStrategy._simulate``
        # (it carries ``fee_schedule``). For Multi reporting we surface
        # gross + commission separately so the table matches other
        # multi pages. Recompute commission for display from the same
        # schedule used by the strategy.
        commission = config.fee_schedule.round_trip(
            trade.entry_price, trade.exit_price, trade.shares
        )
        # Note: trade.pnl already nets commission (BullFlagStrategy
        # tracks commission through the position lifecycle including
        # adds). Use it directly as net_pnl.
        net_pnl = trade.pnl
        gross_pnl = net_pnl + commission
        cost_basis = trade.entry_price * trade.shares
        net_pnl_pct = (
            round(net_pnl / cost_basis, 4) if cost_basis > 0 else 0.0
        )
        # Repurpose MultiTrade.signal_* fields to surface tiebreaker
        # values in the table.
        return MultiTrade(
            ticker=ticker,
            signal_volume=float(fit["rvol"]),  # daily RVOL
            signal_buy_volume=float(fit["float_shares"]) if fit["float_shares"] != float("inf") else 0.0,
            signal_sell_volume=0.0,
            signal_buy_sell_ratio=round(fit["rvol"], 2),  # display = RVOL
            commission=round(commission, 2),
            gross_pnl=round(gross_pnl, 2),
            pattern_name=trade.pattern_name,
            entry_date=trade.entry_date,
            exit_date=trade.exit_date,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            stop_loss=trade.stop_loss,
            shares=trade.shares,
            pnl=round(net_pnl, 2),
            pnl_pct=net_pnl_pct,
            exit_reason=trade.exit_reason,
            entry_ts=trade.entry_ts,
            exit_ts=trade.exit_ts,
        )

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
